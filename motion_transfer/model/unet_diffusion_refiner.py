import os
import torch
import numpy as np
from PIL import Image
import cv2
import mediapipe as mp
import collections
from safetensors.torch import save_file, load_file
from unet_acc import DenseMotion, UNetGenerator, warp_image  # your model



def convert_pth_to_safetensors(pth_path, safe_path):
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"❌ {pth_path} not found")
    print(f"🔄 Converting {pth_path} → {safe_path} ...")
    data = torch.load(pth_path, map_location="cpu")
    tensor_dict = {}

    if isinstance(data, dict) and "dense_motion" in data and "generator" in data:
        for k, v in data["dense_motion"].items():
            if torch.is_tensor(v):
                tensor_dict[f"dense_motion.{k}"] = v.cpu()
        for k, v in data["generator"].items():
            if torch.is_tensor(v):
                tensor_dict[f"generator.{k}"] = v.cpu()
        print(f"✅ Found DenseMotion + Generator weights ({len(tensor_dict)} tensors)")
    else:
        if hasattr(data, "state_dict"):
            data = data.state_dict()
        for k, v in data.items():
            if torch.is_tensor(v):
                tensor_dict[f"generator.{k}"] = v.cpu()
        print(f"⚠️ Only generator weights found ({len(tensor_dict)} tensors)")

    save_file(tensor_dict, safe_path)
    print(f"✅ Saved {safe_path}")
    return safe_path


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
    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.uint8)
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

    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)
    mask = cv2.GaussianBlur(mask, (7, 7), 2)
    return mask.astype(np.float32) / 255.0



class DiffusionRefiner:
    def __init__(self, device, strength=0.30, steps=40, guidance=7.0):
        from diffusers import AutoPipelineForInpainting
        self.device = device
        self.strength = strength
        self.steps = steps
        self.guidance = guidance
        self.pipe = None
        try:
            self.pipe = AutoPipelineForInpainting.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32,
                use_safetensors=True,
                safety_checker=None,
            ).to(device)
            if "cuda" in str(device):
                try: self.pipe.enable_xformers_memory_efficient_attention()
                except Exception: pass
            print("✅ Diffusion refiner ready (SD2 Inpainting).")
        except Exception as e:
            print(f"⚠️ Diffusion refiner unavailable: {e}")

    def refine(self, base_gray, src_gray, mask_gray):
        if self.pipe is None:
            return base_gray

        from PIL import Image
        import cv2
        import numpy as np

        H, W = base_gray.shape
        def to_rgb_pil(gray):
            arr = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
            rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(rgb)

        blended = 0.5 * base_gray + 0.5 * src_gray
        image_pil = to_rgb_pil(blended)
        mask_pil = Image.fromarray((mask_gray * 255).astype(np.uint8))

        try:
            out_img = self.pipe(
                prompt="smooth natural blinking and smiling, realistic pencil sketch face, clean edges",
                negative_prompt="blurry, noisy, flicker, distorted, extra eyes, extra teeth",
                image=image_pil,
                mask_image=mask_pil,
                height=256,
                width=256,
                strength=self.strength,
                guidance_scale=self.guidance,
                num_inference_steps=self.steps,
            ).images[0]
        except Exception as e:
            print(f" Diffusion skipped: {e}")
            return base_gray

        out_gray = cv2.cvtColor(np.array(out_img), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        return np.clip(mask_gray * out_gray + (1 - mask_gray) * base_gray, 0, 1)


def test_single_image(source_image_path, heatmap_dir, output_path="outputs/final_video.avi"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ckpt_dir = "checkpoints"
    pth_path = os.path.join(ckpt_dir, "best.pth")
    safe_path = os.path.join(ckpt_dir, "best.safetensors")
    if not os.path.exists(safe_path) and os.path.exists(pth_path):
        convert_pth_to_safetensors(pth_path, safe_path)

    dense_motion = DenseMotion(kp_channels=68).to(device)
    generator = UNetGenerator(in_channels=4).to(device)
    ckpt = load_file(safe_path)
    dm_state = {k.replace("dense_motion.", ""): v for k, v in ckpt.items() if k.startswith("dense_motion.")}
    gen_state = {k.replace("generator.", ""): v for k, v in ckpt.items() if k.startswith("generator.")}
    dense_motion.load_state_dict(dm_state, strict=False)
    generator.load_state_dict(gen_state, strict=False)
    dense_motion.eval(); generator.eval()
    print("✅ Model loaded from safetensors.")

    refiner = DiffusionRefiner(device)

    src_img = Image.open(source_image_path).convert("L")
    src_np = np.array(src_img, dtype=np.float32) / 255.0
    H, W = src_np.shape
    src_tensor = torch.tensor(src_np).unsqueeze(0).unsqueeze(0).float().to(device)

    keypoints = get_mediapipe_keypoints(source_image_path)
    mask_np = create_eye_mouth_mask((H, W), keypoints)
    mask_tensor = torch.tensor(mask_np).unsqueeze(0).unsqueeze(0).float().to(device)

    heatmap_files = sorted([f for f in os.listdir(heatmap_dir) if f.endswith(".npy")])
    if not heatmap_files:
        print("❌ No heatmaps found.")
        return
    src_kp = torch.tensor(np.load(os.path.join(heatmap_dir, heatmap_files[0]))).permute(2, 0, 1).unsqueeze(0).float().to(device)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), 15, (W, H), False)
    heatmap_buffer = collections.deque(maxlen=20)

    def optical_flow_blend(prev_gray, curr_gray, alpha=0.5):
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        H, W = flow.shape[:2]
        y, x = np.mgrid[0:H, 0:W]
        map_x = (x + flow[..., 0]).astype(np.float32)
        map_y = (y + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(prev_gray, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        return cv2.addWeighted(warped, alpha, curr_gray, 1 - alpha, 0)

    prev_frame = None

    with torch.no_grad():
        for i, h_file in enumerate(heatmap_files):
            drv_np = np.load(os.path.join(heatmap_dir, h_file))
            heatmap_buffer.append(drv_np)
            smoothed_np = np.mean(heatmap_buffer, axis=0).astype(np.float32)

            drv_kp = torch.tensor(smoothed_np).permute(2, 0, 1).unsqueeze(0).float().to(device)

            combined_kp = src_kp.clone()
            for idx in list(range(36, 48)) + list(range(48, 68)):
                combined_kp[:, idx, :, :] = drv_kp[:, idx, :, :]

            flow, occ = dense_motion(src_kp, combined_kp)
            warped_src = torch.clamp(warp_image(src_tensor, flow), 0, 1)
            pred = torch.clamp(generator(torch.cat([warped_src, flow, occ], dim=1)), 0, 1)
            final_frame = pred * mask_tensor + src_tensor * (1 - mask_tensor)
            frame_np = final_frame.cpu().squeeze().numpy()

            if refiner.pipe is not None and (i % 40 == 0):
                frame_np = refiner.refine(frame_np, src_np, mask_np)

            if prev_frame is not None:
                frame_np = optical_flow_blend(prev_frame, frame_np, alpha=0.6)
            prev_frame = frame_np.copy()

            frame_uint8 = (frame_np * 255).astype(np.uint8)
            out.write(frame_uint8)
            if i == 0:
                cv2.imwrite("refined_preview.png", frame_uint8)

    out.release()
    print(f"✅ Final ultra-smooth video saved: {output_path}")



if __name__ == "__main__":
    source_image_path = "C:/Users/Harini/PycharmProjects/style_transfer/motion_transfer/test/60.jpg"
    heatmap_dir = "C:/Users/Harini/PycharmProjects/style_transfer/motion_transfer/test/test_heatmap/"
    output_path = "outputs/final_video.avi"
    test_single_image(source_image_path, heatmap_dir, output_path)
