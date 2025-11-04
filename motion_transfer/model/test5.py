import os
import numpy as np

# ==== CONFIG ====
dataset_root = r"C:\Users\Harini\PycharmProjects\style_transfer\motion_transfer\dataset"
jitter_strength = 1.0  # pixels (adjust between 1.0 and 3.0 for subtle motion)
sigma = 2.0  # for Gaussian heatmap smoothing

# ==== Helper Functions ====
def extract_keypoints(hmap):
    """Extract mean (x, y) for each keypoint channel from a heatmap."""
    kps = []
    for i in range(hmap.shape[2]):
        y, x = np.where(hmap[:, :, i] > 0)
        if len(x) > 0:
            kps.append([np.mean(x), np.mean(y)])
        else:
            kps.append([0, 0])
    return np.array(kps, dtype=np.float32)


def gaussian_heatmaps(points, H, W, sigma=2.0):
    """Recreate Gaussian heatmaps from keypoints."""
    N = points.shape[0]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    heat = np.zeros((H, W, N), dtype=np.float32)
    s2 = 2 * (sigma ** 2)
    for i, (x, y) in enumerate(points):
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        heat[..., i] = np.exp(-d2 / s2)
    return heat


# ==== Main Script ====
persons = sorted([p for p in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, p))])

print(f"🔧 Applying jitter to all 'combined' folders in {dataset_root}...")
print(f"   → Jitter strength: ±{jitter_strength} px")
print(f"   → Gaussian sigma: {sigma}\n")

for person in persons:
    combined_dir = os.path.join(dataset_root, person, "combined")

    if not os.path.exists(combined_dir):
        print(f"⚠️ Skipping {person}: no combined folder found.")
        continue

    npy_files = sorted([f for f in os.listdir(combined_dir) if f.endswith(".npy")])
    if not npy_files:
        print(f"⚠️ Skipping {person}: no .npy files found.")
        continue

    print(f"🧩 Processing {person} — {len(npy_files)} heatmaps")

    for fname in npy_files:
        path = os.path.join(combined_dir, fname)
        heatmap = np.load(path)
        H, W, C = heatmap.shape

        # 1️⃣ Extract keypoints
        kps = extract_keypoints(heatmap)

        # 2️⃣ Apply small random jitter
        jitter = np.random.normal(0, jitter_strength, size=kps.shape)
        jittered_kps = kps + jitter
        jittered_kps = np.clip(jittered_kps, 0, [W - 1, H - 1])  # keep inside image

        # 3️⃣ Regenerate heatmap
        new_heatmap = gaussian_heatmaps(jittered_kps, H, W, sigma=sigma)

        # 4️⃣ Overwrite file
        np.save(path, new_heatmap)

    print(f"✅ Finished: {person}\n")

print("🏁 All 'combined' folders successfully jittered and overwritten.")
