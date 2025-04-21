import torch
from torch import nn
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained VGG16
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False

        self.feature_extractor = create_feature_extractor(
            vgg, return_nodes={
                'features.3': 'conv1_2',
                'features.8': 'conv2_2',
                'features.15': 'conv3_3',
            }
        )

        self.l1_loss = nn.L1Loss()

    def forward(self, X, Y):
        """
        Args:
            X: Generated image tensor in [-1, 1] range (B,C,H,W)
            Y: Target image tensor in [-1, 1] range (B,C,H,W)
        """
        # Normalize to [0,1] range expected by VGG
        X = (X + 1) / 2
        Y = (Y + 1) / 2

        # Extract features - X needs gradient flow
        feat_X = self.feature_extractor(X)

        # Detach Y's computation graph
        with torch.no_grad():
            feat_Y = self.feature_extractor(Y)

        # Compute loss while preserving X's gradients
        loss = 0.0
        for fx, fy in zip(feat_X.values(), feat_Y.values()):
            loss += self.l1_loss(fx, fy)

        return loss