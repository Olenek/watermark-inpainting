import torch
from torch import nn, Tensor
from abc import ABC
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr


class BaseModel(ABC, nn.Module):
    def compute_batch_loss(self, batch, device) -> tuple[Tensor, dict[str, Tensor]]:
        pass

    def compute_stage1_loss(self, batch, device) -> tuple[Tensor, dict[str, Tensor]]:
        pass

    def stage1_parameters(self):
        pass


    @staticmethod
    def denormalize(tensor, device):
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1,3,1,1).to(device)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1,3,1,1).to(device)
        return tensor * std + mean


    def safe_psnr(self, pred, target, device):
        # Denormalize both prediction and target first
        pred = self.denormalize(pred, device)
        target = self.denormalize(target, device)
        # Then clamp to valid range
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        return psnr(pred, target).item()

