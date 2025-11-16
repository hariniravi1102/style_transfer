import os
import cv2
import numpy as np
import torch
import face_alignment
import mediapipe as mp

# ================== CONFIG ==================
source_image_path = r"motion_transfer\test\image\person.jpg"
output_dir = r"motion_transfer\test\single"
os.makedirs(output_dir, exist_ok=True)

target_size = 256
SIGMA = 2.0
NUM_FACE_POINTS = 68
NUM_EXTRAS = 12
NUM_TOTAL = 80

# ================== MODELS ==================
fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
mp_pose = mp.solutions.pose

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

def unit(v, eps=1e-6):
    n = np.linalg.norm(v)
    if n < eps:
        return np.array([0.0, -1.0], dtype=np.float32)
    return (v / n).astype(np.float32)

def gaussian_heatmaps(points, H, W, sigma=2.0):
    N = points.shape[0]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    heat = np.zeros((H, W, N), dtype=np.float32)
    s2 = 2 * (sigma ** 2)
    for i, (x, y) in enumerate(points):
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        heat[..., i] = np.exp(-d2 / s2)
    return heat

# ================== MAIN PROCESS ==================
img_bgr = cv2.imread(source_image_path)
if img_bgr is None:
    raise FileNotFoundError(source_image_path)

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_rgb = resize_with_gradient_padding(img_rgb, target_size)
H, W = img_rgb.shape[:2]

# Face landmarks
fa_out = fa.get_landmarks(img_rgb)
if fa_out is None or len(fa_out) == 0:
    raise RuntimeError("No face detected!")
face68 = fa_out[0][:NUM_FACE_POINTS].astype(np.float32)

# Pose landmarks
with mp_pose.Pose(model_complexity=1, enable_segmentation=False) as pose:
    res_pose = pose.process(img_rgb)
    pose_dict = {}
    if res_pose.pose_landmarks:
        for idx in [7,8,11,12]:
            lm = res_pose.pose_landmarks.landmark[idx]
            pose_dict[idx] = (lm.x * W, lm.y * H)

# Build extras (forehead, head, ears, shoulders)
def build_extras_from_geometry_and_pose(face68, pose_dict, img_shape):
    # copy your function from full code (same as above)
    # returns (12,2) array
    # ...
    return extras12  # placeholder; paste full logic here

extras12 = build_extras_from_geometry_and_pose(face68, pose_dict, img_rgb.shape)
all_points = np.vstack([face68, extras12]).astype(np.float32)  # (80,2)

# Heatmap
heatmap = gaussian_heatmaps(all_points, H, W, sigma=SIGMA)

# Save
np.save(os.path.join(output_dir, "heatmap.npy"), heatmap)

# Optional preview
vis = img_rgb.copy()
for (x, y) in all_points.astype(int):
    cv2.circle(vis, (x, y), 2, (0,255,0), -1)
cv2.imwrite(os.path.join(output_dir, "preview.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

print("✅ Done! Heatmap saved.")
