import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Create Mask for Eyes + Mouth
# -------------------------
def create_eye_mouth_mask(batch_kp, H, W, sigma=10):
    B, N, _ = batch_kp.shape
    device = batch_kp.device
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    yy = yy.float()
    xx = xx.float()
    mask = torch.zeros((B, 1, H, W), device=device)
    for i in range(36, 68):  # eye + mouth indices
        x = batch_kp[:, i, 0].view(B, 1, 1)
        y = batch_kp[:, i, 1].view(B, 1, 1)
        mask += torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


# -------------------------
# Deformation Field Network
# -------------------------
class DeformationFieldNet(nn.Module):
    def __init__(self, kp_channels=68):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(kp_channels * 2, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Global and local branches
        self.flow_global = nn.Conv2d(64, 2, 3, padding=1)
        self.flow_local = nn.Conv2d(64, 2, 3, padding=1)
        self.occ = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, src_kp, drv_kp, mask=None):
        x = torch.cat([src_kp, drv_kp], dim=1)
        feat = self.encoder(x)

        flow_global = self.flow_global(feat)
        flow_local = self.flow_local(feat)
        occ = torch.sigmoid(self.occ(feat))

        # ✅ Resize mask to match flow resolution
        if mask is not None:
            mask = F.interpolate(mask, size=flow_global.shape[2:], mode="bilinear", align_corners=True)

        # Combine global + local deformation
        if mask is not None:
            flow_final = flow_global + flow_local * mask
        else:
            flow_final = flow_global + flow_local

        return flow_final, occ, flow_global, flow_local
