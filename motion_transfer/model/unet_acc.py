import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2


class SketchMotionDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.transform = transform
        self.data = []

        persons = sorted(os.listdir(data_root))
        for person in persons:
            person_dir = os.path.join(data_root, person)
            source_frame = os.path.join(person_dir, f"{person}.jpg")
            frames_dir = os.path.join(person_dir, "frames")
            heatmap_dir = os.path.join(person_dir, "combined")

            frame_files = sorted(os.listdir(frames_dir))
            heatmap_files = sorted(os.listdir(heatmap_dir))

            for f_file, h_file in zip(frame_files, heatmap_files):
                self.data.append({
                    "source_frame": source_frame,
                    "driving_frame": os.path.join(frames_dir, f_file),
                    "driving_heatmap": os.path.join(heatmap_dir, h_file),
                    "source_heatmap": os.path.join(heatmap_dir, heatmap_files[0])  # first frame heatmap
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src_img = Image.open(item["source_frame"]).convert("L")
        drv_img = Image.open(item["driving_frame"]).convert("L")

        src_img = torch.tensor(np.array(src_img)/255.0, dtype=torch.float32).unsqueeze(0)
        drv_img = torch.tensor(np.array(drv_img)/255.0, dtype=torch.float32).unsqueeze(0)

        src_kp = torch.tensor(np.load(item["source_heatmap"]), dtype=torch.float32).permute(2,0,1)
        drv_kp = torch.tensor(np.load(item["driving_heatmap"]), dtype=torch.float32).permute(2,0,1)

        return src_img, drv_img, src_kp, drv_kp


class DenseMotion(nn.Module):
    def __init__(self, kp_channels=68):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(kp_channels*2, 128, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def forward(self, src_kp, drv_kp):
        x = torch.cat([src_kp, drv_kp], dim=1)
        out = self.conv(x)
        flow = out[:, :2, :, :]
        occ = torch.sigmoid(out[:, 2:3, :, :])
        return flow, occ

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(128, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.final(d1)

def warp_image(img, flow):
    B, C, H, W = img.shape
    grid_y, grid_x = torch.meshgrid(torch.linspace(-1,1,H), torch.linspace(-1,1,W))
    grid = torch.stack((grid_x, grid_y),2).unsqueeze(0).repeat(B,1,1,1).to(img.device)
    flow_norm = flow.permute(0,2,3,1) / torch.tensor([W/2, H/2]).to(img.device)
    warped = nn.functional.grid_sample(img, grid + flow_norm, align_corners=True)
    return warped



def save_checkpoint(state, is_best, checkpoint_dir="checkpoints", filename="last.pth", best_filename="best.pth"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        bestpath = os.path.join(checkpoint_dir, best_filename)
        torch.save(state, bestpath)
        print(f" Saved new best checkpoint: {bestpath}")
    else:
        print(f"Saved checkpoint: {filepath}")

def train(data_root, epochs=500, resume_checkpoint="checkpoints/last.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SketchMotionDataset(data_root)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    dense_motion = DenseMotion(kp_channels=68).to(device)
    generator = UNetGenerator(in_channels=4).to(device)

    optimizer = optim.Adam(list(dense_motion.parameters()) + list(generator.parameters()), lr=1e-4)
    criterion = nn.L1Loss()

    start_epoch = 0
    best_loss = float("inf")


    if os.path.exists(resume_checkpoint):
        print(f"Resuming training from {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        dense_motion.load_state_dict(checkpoint["dense_motion"])
        generator.load_state_dict(checkpoint["generator"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0)
        best_loss = checkpoint.get("loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}, last loss = {best_loss:.4f}")
    else:
        print("Starting new training run")

    # ---- Training Loop ----
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0

        for src_img, drv_img, src_kp, drv_kp in dataloader:
            src_img, drv_img = src_img.to(device), drv_img.to(device)
            src_kp, drv_kp = src_kp.to(device), drv_kp.to(device)

            flow, occ = dense_motion(src_kp, drv_kp)
            warped_src = warp_image(src_img, flow)

            unet_input = torch.cat([warped_src, flow, occ], dim=1)
            pred = generator(unet_input)

            loss = criterion(pred, drv_img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        # ---- Save checkpoint ----
        is_best = avg_loss < best_loss
        best_loss = min(avg_loss, best_loss)

        save_checkpoint({
            "epoch": epoch + 1,
            "dense_motion": dense_motion.state_dict(),
            "generator": generator.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": avg_loss,
        }, is_best, checkpoint_dir="checkpoints")

    print("🏁 Training completed.")



def generate_video(data_root, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dense_motion = DenseMotion(kp_channels=68).to(device)
    generator = UNetGenerator(in_channels=4).to(device)

    # Load best checkpoint
    checkpoint = torch.load("checkpoints/best.pth", map_location=device)
    dense_motion.load_state_dict(checkpoint["dense_motion"])
    generator.load_state_dict(checkpoint["generator"])
    dense_motion.eval()
    generator.eval()

    for person in sorted(os.listdir(data_root)):
        person_dir = os.path.join(data_root, person)
        source_frame = os.path.join(person_dir, f"{person}.jpg")
        heatmap_dir = os.path.join(person_dir, "combined")

        src_img = Image.open(source_frame).convert("L")
        src_img = torch.tensor(np.array(src_img)/255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        src_kp = torch.tensor(np.load(os.path.join(heatmap_dir, sorted(os.listdir(heatmap_dir))[0])), dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)

        generated_frames = []
        for h_file in sorted(os.listdir(heatmap_dir)):
            drv_kp = torch.tensor(np.load(os.path.join(heatmap_dir, h_file)), dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)

            flow, occ = dense_motion(src_kp, drv_kp)
            warped_src = warp_image(src_img, flow)
            unet_input = torch.cat([warped_src, flow, occ], dim=1)
            pred = generator(unet_input)
            generated_frames.append(pred.detach().cpu().squeeze().numpy())

        H, W = generated_frames[0].shape
        out_path = os.path.join(output_dir, f"{person}_sketch.avi")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"XVID"), 15, (W,H), False)
        for f in generated_frames:
            out.write((f*255).astype(np.uint8))
        out.release()
        print(f"Video saved: {out_path}")

if __name__ == "__main__":
    data_root = "motion_transfer/dataset/"
    train(data_root, epochs=500, resume_checkpoint="checkpoints/last.pth")  
    generate_video(data_root, output_dir="outputs")
