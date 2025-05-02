import torch.fft
import torch.nn as nn
from einops import rearrange


# --- Helper Modules ---
class scSEBlock(nn.Module):
    def __init__(self, in_channels):
        super(scSEBlock, self).__init__()
        self.channel_excitation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_se = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        chn_se = self.channel_excitation(x)
        spa_se = self.spatial_se(x)
        return x * chn_se + x * spa_se


class SpectralTransform(nn.Module):
    def __init__(self):
        super(SpectralTransform, self).__init__()

    def forward(self, x):
        ffted = torch.fft.fft2(x, norm='ortho')
        ffted = torch.real(ffted)  # keep only real part
        return ffted


class PatchMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.fc2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm2 = nn.LayerNorm(dim)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + residual
        return self.proj(x)


class LocalMLP(nn.Module):
    def __init__(self, dim, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.mlp = PatchMLP(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # Split into 2x2 patches -> rearrange to patch-wise batching
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=self.patch_size, p2=self.patch_size)
        x = x.view(B, -1, self.patch_size * self.patch_size, C)
        x = rearrange(x, 'b n p c -> (b n) c p')  # flatten to apply MLP across patches
        x = x.unsqueeze(-1)  # simulate spatial for conv1d
        x = self.mlp(x)
        x = x.squeeze(-1)
        x = rearrange(x, '(b n) c p -> b c n p', b=B)

        # Reverse rearrangement
        x = rearrange(
            x,
            'b c (h w) (p1 p2) -> b c (h p1) (w p2)',
            h=H // self.patch_size,
            w=W // self.patch_size,
            p1=self.patch_size,
            p2=self.patch_size
        )
        return x


class GlobalMLP(nn.Module):
    def __init__(self, dim, patch_grid=2):
        super().__init__()
        self.grid = patch_grid
        self.mlp = PatchMLP(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # Reshape to grid form
        x = rearrange(x, 'b c (gh h) (gw w) -> b (gh gw) (h w) c', gh=self.grid, gw=self.grid)
        x = rearrange(x, 'b n p c -> (b n) c p')  # flatten to patches
        x = x.unsqueeze(-1)
        x = self.mlp(x)
        x = x.squeeze(-1)
        x = rearrange(x, '(b n) c p -> b c n p', b=B)
        x = rearrange(x, 'b c (gh gw) (h w) -> b c (gh h) (gw w)', gh=self.grid, gw=self.grid, h=H // self.grid,
                      w=W // self.grid)
        return x
