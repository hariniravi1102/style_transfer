import os
import numpy as np
import cv2

# ==== CONFIG ====
single_heatmap_path = "motion_transfer/dataset_single/reference_heatmap/00000.npy"
reference_heatmap_dir = "motion_transfer/dataset_single/reference_heatmap"  # contains 150 npy files
output_dir = "motion_transfer/dataset_single/test_heatmap"
preview_video = "motion_transfer/dataset_single/simulated_motion.mp4"

os.makedirs(output_dir, exist_ok=True)

# ==== Load single heatmap ====
single_heatmap = np.load(single_heatmap_path)
H, W, C = single_heatmap.shape

# Extract keypoints from a heatmap
def extract_keypoints(hmap):
    kps = []
    for i in range(hmap.shape[2]):
        y, x = np.where(hmap[:, :, i] > 0)
        if len(x) > 0:
            kps.append([np.mean(x), np.mean(y)])  # use mean for stability
        else:
            kps.append([0, 0])
    return np.array(kps, dtype=np.float32)

single_kp = extract_keypoints(single_heatmap)

# ==== Load reference motion ====
ref_files = sorted([f for f in os.listdir(reference_heatmap_dir) if f.endswith(".npy")])
ref_heatmaps = [np.load(os.path.join(reference_heatmap_dir, f)) for f in ref_files]
ref_kp_list = [extract_keypoints(hm) for hm in ref_heatmaps]

# ==== Compute motion relative to first reference frame ====
ref_base_kp = ref_kp_list[0]
motion_vectors = [kp - ref_base_kp for kp in ref_kp_list]  # per-frame displacement

# ==== Apply motion to single input keypoints ====
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(preview_video, fourcc, 30, (W, H))

def gaussian_heatmaps(points, H, W, sigma=2.0):
    N = points.shape[0]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    heat = np.zeros((H, W, N), dtype=np.float32)
    s2 = 2 * (sigma ** 2)
    for i, (x, y) in enumerate(points):
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        heat[..., i] = np.exp(-d2 / s2)
    return heat

for frame_idx, displacement in enumerate(motion_vectors):
    moved_kp = single_kp + displacement

    # Generate Gaussian heatmap for all points
    new_heatmap = gaussian_heatmaps(moved_kp, H, W, sigma=2.0)
    np.save(os.path.join(output_dir, f"{frame_idx:05d}.npy"), new_heatmap)

    # Draw preview
    frame_vis = np.zeros((H, W, 3), dtype=np.uint8)
    for (x, y) in moved_kp.astype(int):
        cv2.circle(frame_vis, (x, y), 2, (0, 255, 0), -1)
    video_writer.write(frame_vis)

video_writer.release()

print(f"✅ Simulated motion heatmaps saved in '{output_dir}'")
print(f"🎥 Preview video saved as '{preview_video}'")
