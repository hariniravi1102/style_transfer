import os
import torch
import numpy as np
from PIL import Image
import cv2
import mediapipe as mp
from unet_acc import DenseMotion, UNetGenerator, warp_image




def get_mediapipe_keypoints(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    mp_face = mp.solutions.face_mesh
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1) as mesh:
        res = mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            raise ValueError("❌ No face detected in source image.")
        landmarks = np.array([(p.x * w, p.y * h) for p in res.multi_face_landmarks[0].landmark])
    return landmarks

def create_eye_mouth_mask(image_shape, keypoints):
    """
    Creates a binary (0/1) mask for MediaPipe eyes + mouth regions.
    """
    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.uint8)

    # MediaPipe indices
    left_eye = [33, 133, 160, 159, 158, 157, 173]
    right_eye = [362, 263, 387, 386, 385, 384, 398]
    mouth_outer = list(range(61, 79))
    mouth_inner = list(range(308, 325))

    def fill_region(indices):
        pts = keypoints[indices].astype(np.int32)
        cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)

    fill_region(left_eye)
    fill_region(right_eye)
    fill_region(mouth_outer)
    fill_region(mouth_inner)
    # 🟢 FIX: Dilate slightly and blur to soften transitions
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    mask = cv2.GaussianBlur(mask, (7, 7), 2)

    # Convert to float mask in [0,1]
    mask = mask.astype(np.float32) / 255.0
    return mask




def test_single_image(source_image_path, heatmap_dir, output_path="outputs/test_video_staticmask.avi"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- Load model checkpoint ---
    checkpoint = torch.load("checkpoints/best.pth", map_location=device)
    dense_motion = DenseMotion(kp_channels=68).to(device)
    generator = UNetGenerator(in_channels=4).to(device)
    dense_motion.load_state_dict(checkpoint["dense_motion"])
    generator.load_state_dict(checkpoint["generator"])
    dense_motion.eval()
    generator.eval()

    # --- Load source image ---
    src_img = Image.open(source_image_path).convert("L")
    src_np = np.array(src_img) / 255.0
    H, W = src_np.shape
    src_tensor = torch.tensor(src_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # --- Create eye + mouth mask from MediaPipe ---
    keypoints = get_mediapipe_keypoints(source_image_path)
    mask_np = create_eye_mouth_mask((H, W), keypoints)
    cv2.imwrite("mask_preview.png", (mask_np * 255).astype(np.uint8))
    print("🧩 Saved mask preview as mask_preview.png")

    eye_mouth_mask = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # --- Load keypoint heatmaps ---
    heatmap_files = sorted(os.listdir(heatmap_dir))
    if not heatmap_files:
        print("❌ No heatmaps found.")
        return

    src_kp = torch.tensor(np.load(os.path.join(heatmap_dir, heatmap_files[0])),
                          dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), 15, (W, H), False)

    with torch.no_grad():
        for i, h_file in enumerate(heatmap_files):
            drv_kp = torch.tensor(np.load(os.path.join(heatmap_dir, h_file)),
                                  dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

            # --- Combine keypoints (eyes + mouth dynamic, rest static) ---
            combined_kp = src_kp.clone()
            eye_mouth_indices = list(range(36, 48)) + list(range(48, 68))
            for idx in eye_mouth_indices:
                combined_kp[:, idx, :, :] = drv_kp[:, idx, :, :]

            # --- Compute motion & warp source ---
            flow, occ = dense_motion(src_kp, combined_kp)
            warped_src = warp_image(src_tensor, flow)
            warped_src = torch.clamp(warped_src, 0, 1)

            # --- Generator refinement ---
            input_gen = torch.cat([warped_src, flow, occ], dim=1)
            pred = generator(input_gen)
            pred = torch.clamp(pred, 0, 1)

            # --- ✅ Static background mask (only eyes & mouth animated) ---
            final_frame = pred * eye_mouth_mask + src_tensor * (1 - eye_mouth_mask)
            final_frame = torch.clamp(final_frame, 0, 1)

            # Convert to numpy for saving
            frame_np = final_frame.detach().cpu().squeeze().numpy()
            frame_np = (frame_np * 255).astype(np.uint8)

            if i == 0:
                cv2.imwrite("test_frame_masked.png", frame_np)
                print("🧩 Saved first masked frame as test_frame_masked.png")

            out.write(frame_np)

    out.release()
    print(f"✅ Video saved with static background: {output_path}")


if __name__ == "__main__":
    source_image_path = "C:/Users/Harini/PycharmProjects/style_transfer/motion_transfer/test/60.jpg"
    heatmap_dir = "C:/Users/Harini/PycharmProjects/style_transfer/motion_transfer/test/test_heatmap/"
    output_path = "outputs/test_video_staticmask.avi"

    test_single_image(source_image_path, heatmap_dir, output_path)
