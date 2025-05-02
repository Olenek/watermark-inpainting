import torch
from torch import Tensor

from lib.loss import MaskedL1Loss, VGGPerceptualLoss
from lib.models.base_model import BaseModel
from lib.models.rirci.restoration import RIRCIStage2
from lib.models.slbr.resunet import SLBR


class RIRCIModel(BaseModel):
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
    def __init__(self, gamma=1.5, lambda_1=2., lambda_2=1., lambda_3=3., alpha=.75):
        super().__init__()
        self.stage1 = SLBR(shared_depth=1, blocks=3, long_skip=False)
        self.stage2 = RIRCIStage2()
        self.losses =  {
            'masked_l1': MaskedL1Loss(),
            'vgg': VGGPerceptualLoss(),
            'l1': torch.nn.L1Loss(),
            'bce': torch.nn.BCELoss(),
        }
        self.gamma = gamma
        self.lambda_1, self.lambda_2, self.lambda_3 = lambda_1, lambda_2, lambda_3
        self.alpha = alpha

    def forward(self, J):
        C_b, M_hat = self.stage1(J)
        I_r, I_i, I_hat = self.stage2(J, M_hat, C_b)
        return M_hat, C_b, I_r, I_i, I_hat

    def to(self, *args, **kwargs):
        return_ = super().to(*args, **kwargs)
        for key in self.losses:
            self.losses[key] = self.losses[key].to(*args, **kwargs)

        return return_


    def compute_batch_loss(self, batch, device) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Computes metrics for a single batch.
        Returns: (loss, metrics_dict)
        """
        J = batch['image'].to(device)
        I = batch['target'].to(device)  # Ground truth clean image
        M = batch['mask'].to(device)  # Watermark mask (B,1,H,W)
        W = batch['wm'].to(device)

        C_b = (1 - W) * I
        target_mask = M.squeeze(1)  # (B,H,W)

        # Forward pass
        M_hat, C_b_hat, I_r_hat, I_i_hat, I_hat = self.forward(J)

        M_hat_squeezed = M_hat.squeeze(1)  # Now shape [B, H, W]

        # --- Loss Calculations ---
        L_b = (
                self.lambda_1 * self.losses['masked_l1'](C_b_hat, C_b, M) +
                self.lambda_2 * self.losses['vgg'](C_b_hat, C_b)
        )

        L_r = (
                self.lambda_1 * (
                self.losses['masked_l1'](I_r_hat, I, M * (W > self.alpha)) +
                self.gamma * self.losses['l1'](I_r_hat, I)
        ) +
                self.lambda_2 * self.losses['vgg'](I_r_hat, I)
        )

        L_i = (
                self.lambda_1 * (
                self.losses['masked_l1'](I_i_hat, I, M * (W < self.alpha)) +
                self.gamma * self.losses['l1'](I_i_hat, I)
        ) +
                self.lambda_2 * self.losses['vgg'](I_i_hat, I)
        )

        L_f = (
                self.lambda_1 * (
                self.losses['masked_l1'](I_hat, I, M) +
                self.gamma * self.losses['l1'](I_hat, I)
        ) +
                self.lambda_2 * self.losses['vgg'](I_hat, I)
        )

        L_m = self.lambda_3 * self.losses['bce'](M_hat_squeezed, target_mask)

        batch_loss = L_b + L_r + L_i + L_f + L_m

        loss_components = {
            'L_b': L_b.item(),
            'L_r': L_r.item(),
            'L_i': L_i.item(),
            'L_f': L_f.item(),
            'L_m': L_m.item(),
            'total': batch_loss.item(),
        }

        return batch_loss, loss_components