import torch
import torch.nn as nn

from lib.rirci.glci.glci import GLCI, GLCIStack


class FusionModule(nn.Module):
    """
    Non-local fusion block (simplified version using Conv layers).
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 3, 1)  # Output RGB image
        )

    def forward(self, x):
        return self.conv(x)

class RIRCIStage2(nn.Module):
    """
    RIRCI Stage 2: Background Content Restoration

    Inputs:
        - J: (B, 3, H, W) - original watermarked image
        - M_hat: (B, 1, H, W) - predicted watermark mask
        - C_b: (B, 3, H, W) - predicted watermark-free background component

    Outputs:
        - I_r: (B, 3, H, W) - content-restored image
        - I_i: (B, 3, H, W) - content-imagined image
        - I_hat: (B, 3, H, W) - final fused image
    """
    def __init__(self, glci_channels=64):
        super().__init__()

        self.r_encoder = nn.Sequential(
            nn.Conv2d(4, glci_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.r_glci_0 = GLCI(glci_channels)
        self.r_glci_1 = GLCI(glci_channels)
        self.r_glci_h = GLCIStack(glci_channels, n_blocks=2)  # hidden
        self.r_glci_f = GLCI(glci_channels)  # final block
        self.r_decoder = nn.Sequential(
            nn.Conv2d(glci_channels, glci_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(glci_channels, 3, 1)
        )

        self.i_encoder = nn.Sequential(
            nn.Conv2d(4, glci_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.i_glci_1 = GLCI(glci_channels)
        self.i_glci_2 = GLCI(glci_channels)
        self.i_glci_h = GLCIStack(glci_channels, n_blocks=2)
        self.i_glci_o = GLCI(glci_channels)
        self.i_decoder = nn.Sequential(
            nn.Conv2d(glci_channels, glci_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(glci_channels, 3, 1)
        )

        self.fusion = FusionModule(in_channels=3 + 3 + 1)  # I_r, I_i, M_hat

    def forward(self, J, M_hat, C_b):
        # Content Restoration Sub-network
        cb_mask = torch.cat([C_b, M_hat], dim=1)  # (B, 4, H, W)
        x = self.r_encoder(cb_mask)
        r_0 = self.r_glci_0(x)
        r_1 = self.r_glci_1(r_0)
        r_h = self.r_glci_h(r_1)
        r_f = self.r_glci_f(r_h) + r_1 # Skip from R. GLCI_2 → R. GLCI_F
        I_r = self.r_decoder(r_f)

        # Content Imagination Sub-network
        J_masked = (1 - M_hat) * J
        input_imagine = torch.cat([J_masked, M_hat], dim=1)
        x = self.i_encoder(input_imagine)
        i_0 = self.i_glci_1(x) + r_0 # Skip from R. GLCI_1 → I. GLCI_1
        i_1 = self.i_glci_2(i_0) + r_1 # Skip from R. GLCI_1 → I. GLCI_1
        i_h = self.i_glci_h(i_1)
        i_o = self.i_glci_o(i_h) + i_1 # Skip from R. GLCI_1 → I. GLCI_1
        I_i = self.i_decoder(i_o)

        # Fusion
        fusion_input = torch.cat([I_r, I_i, M_hat], dim=1)
        I_hat = self.fusion(fusion_input)

        return I_r, I_i, I_hat
