import torch.nn as nn

from lib.rirci.exclusion import RIRCIStage1
from lib.rirci.restoration import RIRCIStage2
from lib.slbr.resunet import SLBR


class RIRCIModel(nn.Module):
    """
    Full RIRCI Visible Watermark Removal Model (Stage 1 + Stage 2)

    Input:
        - J: (B, 3, H, W) - Watermarked image

    Outputs:
        - M_hat: (B, 1, H, W) - Predicted watermark mask
        - C_b:   (B, 3, H, W) - Estimated clean background component
        - I_r:   (B, 3, H, W) - Restored image (from intrinsic background)
        - I_i:   (B, 3, H, W) - Imagined image (from contextual info)
        - I_hat: (B, 3, H, W) - Final fused clean image
    """
    def __init__(self):
        super().__init__()
        self.stage1 = SLBR(shared_depth=1, blocks=3, long_skip=False)
        self.stage2 = RIRCIStage2()

    def forward(self, J):
        C_b, M_hat = self.stage1(J)
        I_r, I_i, I_hat = self.stage2(J, M_hat, C_b)
        return M_hat, C_b, I_r, I_i, I_hat
