import torch
import torch.nn as nn

from lib.models.rirci.glci.glci import GLCI, GLCIStack


class SimpleFusionModule(nn.Module):
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


# class NonLocalBlock(nn.Module):
#     def __init__(self, in_channels, inter_channels=None):
#         super().__init__()
#         self.in_channels = in_channels
#         self.inter_channels = inter_channels if inter_channels else in_channels // 2
#
#         # Query, Key, Value transformations
#         self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
#         self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
#         self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
#
#         # Output transformation
#         self.W = nn.Sequential(
#             nn.Conv2d(self.inter_channels, in_channels, kernel_size=1),
#             nn.BatchNorm2d(in_channels)  # Add BatchNorm after the final conv
#         )
#         nn.init.constant_(self.W[0].weight, 0)
#         nn.init.constant_(self.W[0].bias, 0)
#
#     def forward(self, x):
#         residual = x
#         batch_size = x.size(0)
#
#         # Transformations
#         g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # (B, C', H*W)
#         theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
#         phi_x = self.phi(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
#
#         # Affinity and softmax
#         f = torch.matmul(theta_x, phi_x)  # (B, H*W, H*W)
#         f_softmax = torch.softmax(f, dim=-1)
#
#         # Weighted sum
#         y = torch.matmul(f_softmax, g_x)  # (B, C', H*W)
#         y = y.view(batch_size, self.inter_channels, *x.shape[2:])
#
#         # Residual + normalization
#         z = self.W(y) + residual
#         return z
#
#
# class FusionModule(nn.Module):
#     def __init__(self, in_channels=7, out_channels=3):
#         super().__init__()
#         # Conv 3x3 -> BatchNorm -> Non-local -> Conv 1x1 -> BatchNorm -> Non-local
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.non_local1 = NonLocalBlock(in_channels)
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.non_local2 = NonLocalBlock(out_channels)
#
#     def forward(self, x):
#         x = self.conv1(x)  # (B,7,H,W) → (B,7,H,W)
#         x = self.non_local1(x)  # (B,7,H,W) → (B,7,H,W)
#         x = self.conv2(x)  # (B,7,H,W) → (B,3,H,W)
#         x = self.non_local2(x)  # (B,3,H,W) → (B,3,H,W)
#         return x

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        inter_channels = in_channels // 2  # Common practice is to use half channels

        # Transformations
        self.g = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.theta = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, inter_channels, kernel_size=1)

        # Output transformation with residual
        self.W = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )

        # Proper initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, -1, x.size(2) * x.size(3))  # (B, C', H*W)
        theta_x = self.theta(x).view(batch_size, -1, x.size(2) * x.size(3))
        phi_x = self.phi(x).view(batch_size, -1, x.size(2) * x.size(3)).permute(0, 2, 1)

        f = torch.matmul(theta_x, phi_x)  # (B, H*W, H*W)
        f_div_C = torch.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)  # (B, C', H*W)
        y = y.view(batch_size, -1, x.size(2), x.size(3))

        z = self.W(y)
        return z + residual


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Proper initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.conv(x)


class FusionModule(nn.Module):
    def __init__(self, in_channels=7, out_channels=3):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, in_channels)
        self.non_local1 = NonLocalBlock(in_channels)

        self.conv2 = ConvBlock(in_channels, out_channels)
        self.non_local2 = NonLocalBlock(out_channels)


    def forward(self, x):
        x = self.conv1(x)  # (B,7,H,W) → (B,7,H,W)
        x = self.non_local1(x)  # (B,7,H,W) → (B,7,H,W)
        x = self.conv2(x)  # (B,7,H,W) → (B,3,H,W)
        x = self.non_local2(x)  # (B,3,H,W) → (B,3,H,W)
        return x

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

    def __init__(self, glci_channels=64, simple=False):
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

        if simple:
            self.fusion = SimpleFusionModule(in_channels=3 + 3 + 1)
        else:
            self.fusion = FusionModule(in_channels=3 + 3 + 1)  # I_r, I_i, M_hat

    def forward(self, J, M_hat, C_b):
        # Content Restoration Sub-network
        cb_mask = torch.cat([C_b, M_hat], dim=1)  # (B, 4, H, W)
        x = self.r_encoder(cb_mask)
        r_0 = self.r_glci_0(x)
        r_1 = self.r_glci_1(r_0)
        r_h = self.r_glci_h(r_1)
        r_f = self.r_glci_f(r_h) + r_1  # Skip from R. GLCI_2 → R. GLCI_F
        I_r = self.r_decoder(r_f)

        # Content Imagination Sub-network
        J_masked = (1 - M_hat) * J
        input_imagine = torch.cat([J_masked, M_hat], dim=1)
        x = self.i_encoder(input_imagine)
        i_0 = self.i_glci_1(x) + r_0  # Skip from R. GLCI_1 → I. GLCI_1
        i_1 = self.i_glci_2(i_0) + r_1  # Skip from R. GLCI_1 → I. GLCI_1
        i_h = self.i_glci_h(i_1)
        i_o = self.i_glci_o(i_h) + i_1  # Skip from R. GLCI_1 → I. GLCI_1
        I_i = self.i_decoder(i_o)

        # Fusion
        fusion_input = torch.cat([I_r, I_i, M_hat], dim=1)
        I_hat = self.fusion(fusion_input)

        return I_r, I_i, I_hat
