import os
import numpy as np

heatmap_dir = "motion_transfer/dataset_single/test_heatmap"
smoothed_dir = "motion_transfer/dataset_single/smoothed_heatmaps"
os.makedirs(smoothed_dir, exist_ok=True)

# Step 1: Load existing heatmaps and extract keypoints
heatmap_files = sorted([f for f in os.listdir(heatmap_dir) if f.endswith(".npy")])
kp_list = [np.load(os.path.join(heatmap_dir, f)) for f in heatmap_files]

# Extract keypoints from each heatmap
def extract_kpoints(hm):
    kps = []
    for i in range(hm.shape[2]):
        y, x = np.where(hm[:, :, i] > 0)
        if len(x) > 0:
            kps.append([np.mean(x), np.mean(y)])
        else:
            kps.append([0, 0])
    return np.array(kps, dtype=np.float32)

kp_list = [extract_kpoints(hm) for hm in kp_list]

# Step 2: Apply temporal smoothing
def temporal_smoothing(kp_list, alpha=0.7):
    smoothed = [kp_list[0].copy()]
    for i in range(1, len(kp_list)):
        new_kp = alpha * smoothed[-1] + (1 - alpha) * kp_list[i]
        smoothed.append(new_kp)
    return smoothed

smoothed_kp_list = temporal_smoothing(kp_list, alpha=0.7)

# Step 3: Recompute heatmaps with same dimensions
H, W, C = np.load(os.path.join(heatmap_dir, heatmap_files[0])).shape

def gaussian_heatmaps(points, H, W, sigma=2.0):
    N = points.shape[0]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    heat = np.zeros((H, W, N), dtype=np.float32)
    s2 = 2 * (sigma ** 2)
    for i, (x, y) in enumerate(points):
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        heat[..., i] = np.exp(-d2 / s2)
    return heat

for idx, kp in enumerate(smoothed_kp_list):
    new_hm = gaussian_heatmaps(kp, H, W, sigma=2.0)
    np.save(os.path.join(smoothed_dir, heatmap_files[idx]), new_hm)
