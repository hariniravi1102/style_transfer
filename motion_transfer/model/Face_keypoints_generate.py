import os
import glob
import cv2
import numpy as np
import torch
import face_alignment
from multiprocessing import Pool, cpu_count

# ================== CONFIG ==================
sources_dir = r"motion_transfer\new_dataset\image"
videos_dir  = r"motion_transfer\new_dataset\videos"
output_root = r"motion_transfer\dataset"

num_workers = min(cpu_count(), 4)
target_size = 256
SIGMA = 2.0

NUM_FACE_POINTS = 68  # only 68 now

# Temporal smoothing (exponential moving average)
SMOOTH_ALPHA = 0.7  # higher => smoother (less jitter), but slower to react

# ================== MODELS ==================
fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# ================== UTILS ==================
def resize_with_gradient_padding(img, target_size):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    tl = np.mean(img[0:5, 0:5, :], axis=(0,1))
    tr = np.mean(img[0:5, -5:, :], axis=(0,1))
    grad = np.linspace(tl, tr, target_size)
    bg = np.tile(grad, (target_size, 1, 1)).astype(np.uint8)
    y0 = (target_size - h) // 2
    x0 = (target_size - w) // 2
    bg[y0:y0+h, x0:x0+w] = img
    return bg

def clip_xy(x, y, w, h):
    return float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1))

def ema(prev_pts, curr_pts, alpha=SMOOTH_ALPHA):
    if prev_pts is None or prev_pts.shape != curr_pts.shape:
        return curr_pts
    return alpha * curr_pts + (1.0 - alpha) * prev_pts

def gaussian_heatmaps(points, H, W, sigma=2.0):
    N = points.shape[0]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    heat = np.zeros((H, W, N), dtype=np.float32)
    s2 = 2 * (sigma ** 2)
    for i, (x, y) in enumerate(points):
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        heat[..., i] = np.exp(-d2 / s2)
    return heat

# ================== MAIN PER-PERSON ==================
def process_person(person_name):
    source_path = os.path.join(sources_dir, f"{person_name}.png")
    video_path  = os.path.join(videos_dir,  f"{person_name}.mp4")
    if not (os.path.isfile(source_path) and os.path.isfile(video_path)):
        print(f"❌ Missing files for {person_name}")
        return

    print(f"▶ Processing {person_name}...")
    person_root  = os.path.join(output_root, person_name)
    frames_dir = os.path.join(output_root, person_name, "frames")
    combined_dir = os.path.join(output_root, person_name, "combined")
    keypoints_preview_dir = os.path.join(output_root, person_name, "keypoints_preview")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)
    os.makedirs(keypoints_preview_dir, exist_ok=True)

    # Save resized source image copy (optional)
    src_img = cv2.imread(source_path)
    if src_img is not None:
        src_ref = resize_with_gradient_padding(src_img, target_size)
        cv2.imwrite(os.path.join(person_root, f"{person_name}.jpg"), src_ref)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    prev_points = None

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        # resize/pad to target
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = resize_with_gradient_padding(frame_rgb, target_size)
        H, W = frame_rgb.shape[:2]

        # 68 face landmarks
        fa_out = fa.get_landmarks(frame_rgb)
        if fa_out is None or len(fa_out) == 0 or fa_out[0].shape[0] < NUM_FACE_POINTS:
            frame_idx += 1
            continue
        face68 = fa_out[0][:NUM_FACE_POINTS].astype(np.float32)

        # Smooth for temporal stability
        face68 = ema(prev_points, face68, alpha=SMOOTH_ALPHA)
        prev_points = face68.copy()

        # Heatmaps
        heatmap = gaussian_heatmaps(face68, H, W, sigma=SIGMA)

        # Save preview + .npy
        vis = frame_rgb.copy()
        for (x, y) in face68.astype(int):
            cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

        fname = f"{frame_idx:05d}"
        frame_file = f"{frame_idx:05d}.jpg"
        cv2.imwrite(os.path.join(frames_dir, frame_file), frame_rgb)
        #cv2.imwrite(os.path.join(keypoints_preview_dir, f"{fname}.png"), vis_bgr)
        #np.save(os.path.join(combined_dir, f"{fname}.npy"), heatmap)

        frame_idx += 1

    cap.release()
    print(f"✅ Done: {person_name} | Frames: {frame_idx} | Points/frame: {NUM_FACE_POINTS}")

# ================== RUN ==================
if __name__ == "__main__":
    people = [os.path.splitext(os.path.basename(p))[0]
              for p in glob.glob(os.path.join(sources_dir, "*.png"))]
    with Pool(num_workers) as p:
        p.map(process_person, people)
