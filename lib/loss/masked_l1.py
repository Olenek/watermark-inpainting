import torch
import torch.nn as nn


class MaskedL1Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps  # Small value to prevent division by zero
        self.l1_loss = nn.L1Loss(reduction='none')

    def forward(self, X, Y, M):
        """
        NaN-safe masked L1 loss.

        Args:
            X: Generated image (B,C,H,W)
            Y: Target image (B,C,H,W)
            M: Binary mask (B,1,H,W), 1=watermark region, 0=background
        Returns:
            loss: Masked L1 loss value with NaN handling
        """
        # Calculate pixel-wise L1 differences
        l1_diff = self.l1_loss(X, Y)  # Shape: (B,C,H,W)

        # Expand mask to match channel dimension if needed
        if M.size(1) != X.size(1):
            M = M.expand_as(X)

        # Create NaN mask (1 where inputs are valid)
        valid_mask = (~torch.isnan(X)) & (~torch.isnan(Y))

        # Combined mask (apply both watermark mask and NaN mask)
        combined_mask = M * valid_mask.float()

        # Sum of absolute differences only in valid regions
        sum_diff = torch.sum(combined_mask * l1_diff)

        # Count of valid elements (avoid division by zero)
        valid_count = torch.sum(combined_mask) + self.eps

        return sum_diff / valid_count