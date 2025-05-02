from torch import nn, Tensor
from abc import ABC

class BaseModel(ABC, nn.Module):
    def compute_batch_loss(self, batch, device) -> tuple[Tensor, dict[str, Tensor]]:
        pass
