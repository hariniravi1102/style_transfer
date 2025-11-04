import os
import torch
import numpy as np
import cv2
from unet_acc import DenseMotion  # your existing DenseMotion class


# -----------------------------
# CONFIG
# -----------------------------
heatmap_dir = r"C:\Users\Harini\PycharmProjects\style_transfer\motion_transfer\dataset_single\reference_heatmap"
checkpoint_path = "checkpoints/best.pth"
output_dir = "outputs/flow_debug/"
os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD MODEL
# -----------------------------
print("🔄 Loading DenseMotion checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location=device)
dense_motion = DenseMotion(kp_channels=68).to(device)
dense_motion.load_state_dict(checkpoint["dense_motion"])
dense_motion.eval()

# -----------------------------
# LOAD HEATMAPS
# -----------------------------
heatmap_files = sorted([f for f in os.listdir(heatmap_dir) if f.endswith(".npy")])
if len(heatmap_files) < 2:
    print("❌ Not enough heatmap frames to compute motion!")
    exit()

src_kp_np = np.load(os.path.join(heatmap_dir, heatmap_files[0]))
src_kp = torch.tensor(src_kp_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

flow_means, flow_maxs = [], []

# -----------------------------
# MAIN LOOP
# -----------------------------
with torch.no_grad():
    for i, file in enumerate(heatmap_files):
        drv_kp_np = np.load(os.path.join(heatmap_dir, file))
        drv_kp = torch.tensor(drv_kp_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

        flow, occ = dense_motion(src_kp, drv_kp)
        flow_mag = torch.norm(flow, dim=1)  # (B, H, W)
        mean_val = flow_mag.mean().item()
        max_val = flow_mag.max().item()

        flow_means.append(mean_val)
        flow_maxs.append(max_val)

        # Visualize every 10th frame
        if i % 10 == 0:
            flow_np = flow[0].detach().cpu().numpy()
            flow_x, flow_y = flow_np[0], flow_np[1]
            mag, ang = cv2.cartToPolar(flow_x, flow_y, angleInDegrees=True)
            mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            hsv = np.zeros((mag.shape[0], mag.shape[1], 3), dtype=np.uint8)
            hsv[..., 0] = ang / 2  # hue = direction
            hsv[..., 1] = 255
            hsv[..., 2] = mag_norm.astype(np.uint8)
            flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite(os.path.join(output_dir, f"flow_vis_{i:03d}.png"), flow_rgb)

        # Print progress
        if i % 20 == 0:
            print(f"Frame {i:03d}: flow_mean={mean_val:.6f}, flow_max={max_val:.6f}")

# -----------------------------
# SUMMARY
# -----------------------------
flow_means = np.array(flow_means)
flow_maxs = np.array(flow_maxs)
print("\n===== FLOW SUMMARY =====")
print(f"Total frames analyzed: {len(flow_means)}")
print(f"Mean(flow_mean): {flow_means.mean():.6f}")
print(f"Mean(flow_max):  {flow_maxs.mean():.6f}")
print(f"Max of all flow_max: {flow_maxs.max():.6f}")
print(f"Saved flow visualizations every 10th frame in: {output_dir}")
print("=========================\n")

if flow_maxs.max() < 0.02:
    print("⚠️ Flow magnitude extremely low — likely no motion in heatmaps.")
elif flow_maxs.max() < 0.05:
    print("⚠️ Very weak motion detected — blinking/smile signals too small.")
else:
    print("✅ Flow magnitude OK — DenseMotion is generating visible motion.")
