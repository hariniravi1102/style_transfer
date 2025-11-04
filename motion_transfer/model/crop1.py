import cv2
import os
import torch
import face_alignment
import numpy as np
from multiprocessing import cpu_count
from test_acc import test_single_image
import streamlit as st
import tempfile
import mediapipe as mp

# ==== CONFIG ====
reference_heatmap_dir = "C:/Users/Harini/PycharmProjects/style_transfer/motion_transfer/dataset_single/reference_heatmap"
output_dir = "C:/Users/Harini/PycharmProjects/style_transfer/motion_transfer/dataset_single/test_heatmap"
final_output = "C:/Users/Harini/PycharmProjects/style_transfer/motion_transfer/outputs/final_result.mp4"
os.makedirs(output_dir, exist_ok=True)

num_workers = min(cpu_count(), 4)
target_size = 256
SIGMA = 2.0
NUM_FACE_POINTS = 68

# ==== MODELS ====
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
                                  device="cuda" if torch.cuda.is_available() else "cpu")
mp_face_detection = mp.solutions.face_detection
detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)


# ==== HELPERS ====
def gaussian_heatmaps(points, H, W, sigma=2.0):
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    heatmaps = np.exp(-((xx[..., None] - points[:, 0]) ** 2 + (yy[..., None] - points[:, 1]) ** 2) / (2 * sigma ** 2))
    return heatmaps.astype(np.float32)


def extract_keypoints(hmap):
    kps = []
    for i in range(hmap.shape[2]):
        y, x = np.where(hmap[:, :, i] > 0)
        if len(x) > 0:
            kps.append([np.mean(x), np.mean(y)])
        else:
            kps.append([0, 0])
    return np.array(kps, dtype=np.float32)


def trim_video(input_path, output_path, max_seconds=7):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("❌ Error opening video")
        return False
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(total_frames, fps * max_seconds)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        count += 1
    cap.release()
    out.release()
    return True


def crop_head_with_bg(img_rgb, target_size=256, margin_top=0.6, margin_sides=0.3, margin_bottom=0.4):
    """
    Crop face+ears+neck with background, with extra margin above the head.
    Keeps proportions, resizes safely to target_size,
    and pads with a gradient background estimated from top corners.
    """
    ih, iw, _ = img_rgb.shape
    results = detector.process(img_rgb)
    if not results.detections:
        return None

    det = results.detections[0]
    bbox = det.location_data.relative_bounding_box
    x1 = int(bbox.xmin * iw)
    y1 = int(bbox.ymin * ih)
    w = int(bbox.width * iw)
    h = int(bbox.height * ih)

    # Expand differently for each side
    x1 = max(0, int(x1 - w * margin_sides))
    x2 = min(iw, int(x1 + w * (1 + 2 * margin_sides)))
    y1 = max(0, int(y1 - h * margin_top))        # extra space above hair
    y2 = min(ih, int(y1 + h * (1 + margin_bottom + margin_top)))

    cropped = img_rgb[y1:y2, x1:x2]

    # Resize while keeping aspect ratio
    ch, cw = cropped.shape[:2]
    scale = target_size / max(ch, cw)
    new_w, new_h = int(cw * scale), int(ch * scale)
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # === Gradient padding instead of white ===
    tl = np.mean(resized[0:5, 0:5, :], axis=(0, 1))    # top-left avg color
    tr = np.mean(resized[0:5, -5:, :], axis=(0, 1))    # top-right avg color
    grad = np.linspace(tl, tr, target_size)
    canvas = np.tile(grad, (target_size, 1, 1)).astype(np.uint8)

    # Place resized crop on gradient background
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas

# ==== Streamlit UI ====
st.title("Sketch to Live")
src_img = st.file_uploader("Upload face sketch high resolution", type=["jpg", "png"])
cropped_head = None

if src_img is not None:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(src_img.read())
        temp_img_path = tmp.name

    img_bgr = cv2.imread(temp_img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ih, iw, _ = img_rgb.shape

    if ih < target_size or iw < target_size:
        st.warning(f"⚠️ Image too small ({iw}×{ih}). Please upload an image larger than {target_size}×{target_size}.")
    else:
        cropped_head = crop_head_with_bg(img_rgb, target_size=target_size)
        if cropped_head is None:
            st.warning("⚠️ No face detected. Try another image.")
        else:
            st.subheader("Face Preview")
            st.image(cropped_head, use_column_width=True)
            cv2.imwrite("cropped_head.png", cv2.cvtColor(cropped_head, cv2.COLOR_RGB2BGR))


# ==== Motion Transfer with Progress Bar ====
if st.button("Lively Sketch"):
    if cropped_head is None:
        st.error("❌ Please upload a sufficiently large image and crop the head first.")
    else:
        progress_text = st.empty()
        progress_bar = st.progress(0)
        frame_preview = st.empty()
        progress_text.text("⏳ Processing...")

        # Ensure cropped head is exactly 256x256
        #cropped_head = cv2.resize(cropped_head, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

        H, W = cropped_head.shape[:2]

        # Face landmarks
        fa_out = fa.get_landmarks(cropped_head)
        if fa_out is None or len(fa_out) == 0:
            st.error("❌ No face detected. Upload another image")
        else:
            face68 = fa_out[0].astype(np.float32)
            single_heatmap = gaussian_heatmaps(face68, H, W, sigma=SIGMA)
            single_kp = face68

            # Reference heatmaps
            ref_files = sorted([f for f in os.listdir(reference_heatmap_dir) if f.endswith(".npy")])
            if len(ref_files) == 0:
                st.error("❌ No reference heatmaps found!")
            else:
                ref_heatmaps = [np.load(os.path.join(reference_heatmap_dir, f)) for f in ref_files]
                ref_kp_list = [extract_keypoints(hm) for hm in ref_heatmaps]
                ref_base_kp = ref_kp_list[0]
                motion_vectors = [kp - ref_base_kp for kp in ref_kp_list]

                os.makedirs(output_dir, exist_ok=True)
                total_frames = len(motion_vectors)

                # Generate heatmaps
                for frame_idx, displacement in enumerate(motion_vectors):
                    moved_kp = single_kp + displacement
                    new_heatmap = gaussian_heatmaps(moved_kp, H, W, sigma=SIGMA)
                    np.save(os.path.join(output_dir, f"{frame_idx:05d}.npy"), new_heatmap)

                    # Live preview
                    frame_preview.image(cropped_head, width=128)
                    progress_bar.progress(int((frame_idx + 1)/total_frames * 100))

                # Call the motion transfer model
                temp_img_path = "cropped_head.png"
                cv2.imwrite(temp_img_path, cv2.cvtColor(cropped_head, cv2.COLOR_RGB2BGR))
                test_single_image(temp_img_path, output_dir, final_output)

                # Trim video
                trimmed_output = "trimmed_result.mp4"
                trim_video(final_output, trimmed_output, max_seconds=7)

                progress_bar.progress(100)
                progress_text.text("✅ Done!")
                frame_preview.empty()

                st.success("Sketch to live ready!")
                with open(trimmed_output, "rb") as f:
                    st.download_button("⬇️ Download Result Video", f, file_name="Sketch.mp4")


# ==== Footer ====
st.markdown("""
<div style='position: fixed; bottom: 10px; right: 10px; color: gray; font-size: 12px;'>
    Inspired by <b>FOMM (First Order Motion Model)</b>
</div>
""", unsafe_allow_html=True)
