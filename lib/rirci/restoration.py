import torch
import torch.nn as nn

from lib.rirci.glci import GLCIBlock


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


class RestorationBranch(nn.Module):
    """
    Content Restoration Sub-network
    Input: torch.Tensor of shape (B, 4, H, W) [C_b || M_hat]
    Output: torch.Tensor of shape (B, 3, H, W) - restored image I_r
    """
    def __init__(self, glci_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, glci_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.glci = GLCIBlock(glci_channels)
        self.decoder = nn.Sequential(
            nn.Conv2d(glci_channels, glci_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(glci_channels, 3, 1)  # Output RGB
        )

    def forward(self, cb_mask_concat):
        x = self.encoder(cb_mask_concat)
        x = self.glci(x)
        return self.decoder(x)


class ImaginationBranch(nn.Module):
    """
    Content Imagination Sub-network
    Input: torch.Tensor of shape (B, 4, H, W) [(1 - M_hat) * J || M_hat]
    Output: torch.Tensor of shape (B, 3, H, W) - imagined image I_i
    """
    def __init__(self, glci_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, glci_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.glci = GLCIBlock(glci_channels)
        self.decoder = nn.Sequential(
            nn.Conv2d(glci_channels, glci_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(glci_channels, 3, 1)
        )

    def forward(self, masked_input):
        x = self.encoder(masked_input)
        x = self.glci(x)
        return self.decoder(x)


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
        self.restoration = RestorationBranch(glci_channels)
        self.imagination = ImaginationBranch(glci_channels)
        self.fusion = FusionModule(in_channels=3 + 3 + 1)  # I_r, I_i, M_hat

    def forward(self, J, M_hat, C_b):
        # Content Restoration Sub-network
        cb_mask = torch.cat([C_b, M_hat], dim=1)  # (B, 4, H, W)
        I_r = self.restoration(cb_mask)

        # Content Imagination Sub-network
        J_masked = (1 - M_hat) * J
        input_imagine = torch.cat([J_masked, M_hat], dim=1)
        I_i = self.imagination(input_imagine)

        # Fusion
        fusion_input = torch.cat([I_r, I_i, M_hat], dim=1)
        I_hat = self.fusion(fusion_input)

        return I_r, I_i, I_hat
