import torch
import torch.nn as nn

from lib.rirci.glci.base import GlobalMLP, LocalMLP, scSEBlock, SpectralTransform


class GLCI(nn.Module):
    def __init__(self, in_channels):
        super(GLCI, self).__init__()
        self.global_mlp = GlobalMLP(in_channels)
        self.local_mlp = LocalMLP(in_channels)
        self.scse = scSEBlock(in_channels)
        self.spectral = SpectralTransform()
        self.output_proj = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        local_in = x
        global_in = x

        # Global-to-Global skip connection
        g2g = self.global_mlp(global_in) + global_in

        # Local-to-Global skip connection
        l2g = self.spectral(local_in)
        g_out = g2g + l2g

        # Local-to-Local skip connection
        l2l = self.local_mlp(local_in) + local_in

        # Global-to-Local skip connection
        g2l = self.scse(global_in)
        l_out = l2l + g2l

        # Concatenation of global + local features
        out = torch.cat([g_out, l_out], dim=1)  # [B, 2C, H, W]

        # Final 1x1 Conv for output projection (prepares for stacking)
        return self.output_proj(out)           # [B, C, H, W]




class GLCIStack(nn.Module):
    def __init__(self, in_channels, n_blocks=3):
        super().__init__()
        self.blocks = nn.Sequential(*[GLCI(in_channels) for _ in range(n_blocks)])

    def forward(self, x):
        return self.blocks(x)
