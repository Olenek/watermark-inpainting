import torch.nn as nn
from abc import ABC
from torch import Tensor


class SplitNet(ABC, nn.Module):
    def forward(self, x) -> tuple[tuple[Tensor, Tensor], Tensor, Tensor]:
        pass