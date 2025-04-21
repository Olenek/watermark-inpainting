import torch.nn as nn
from lib.rirci.base import DownBlock, UpBlock, SMRBranch, MBEBranch, ConvBlock


class RIRCIStage1(nn.Module):
    """
    Stage 1 of RIRCI: Watermark Component Exclusion

    Input:
        - J: (B, 3, H, W) - Watermarked image

    Output:
        - M_hat: (B, 1, H, W) - Predicted watermark mask
        - C_b_hat: (B, 3, H, W) - Background component (J - M_hat * C_w_hat)
    """
    def __init__(self):
        super().__init__()

        # Encoder
        self.inc = ConvBlock(3, 64)  # no pooling
        self.enc1 = DownBlock(64, 128)   # 256 → 128
        self.enc2 = DownBlock(128, 256)  # 128 → 64
        self.enc3 = DownBlock(256, 512)  # 64 → 32
        self.enc4 = DownBlock(512, 1024) # 32 -> 16

        # Decoder
        self.up4 = UpBlock(1024, 512)  # 16 → 32
        self.up3 = UpBlock(512, 256)   # 32 → 64
        self.up2 = UpBlock(256, 128)   # 64 → 128
        self.up1 = UpBlock(128, 64)     # 128 → 256

        # Output branches
        self.smr = SMRBranch(64)  # Predict mask
        self.mbe = MBEBranch(64)  # Predict watermark component

    def forward(self, J):
        # Encoder
        x1 = self.inc(J)  # 64, 256x256
        x2 = self.enc1(x1) # 128, 128x128
        x3 = self.enc2(x2) # 256, 64x64
        x4 = self.enc3(x3) # 512, 32x32
        x5 = self.enc4(x4)  # 1024, 16x16

        # Decoder
        x = self.up4(x5, x4)  # 512, 32x32
        x = self.up3(x, x3)  # 256, 64x64
        x = self.up2(x, x2)  # 128, 128x128
        x = self.up1(x, x1)  # 64, 256x256

        # Branch outputs
        M_hat = self.smr(x)            # (B, 1, H, W)
        C_w_hat = self.mbe(x)          # (B, 3, H, W)
        C_b_hat = J - M_hat * C_w_hat  # element-wise mask blending

        return M_hat, C_b_hat
