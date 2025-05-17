from typing import cast

from torch import Tensor

from lib.models.rirci import RIRCIModel
from lib.models.splitnet_interface import SplitNet


class InferenceWrapper:
    def __init__(self, model: RIRCIModel | SplitNet):
        if isinstance(model, RIRCIModel):
            self.mode = 'rirci'
            self.model = cast(RIRCIModel, model)

        else:
            self.mode = 'splitnet'
            self.model = cast(SplitNet, model)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if self.mode == 'rirci':
            M_hat, _, _, _, I_hat = self.model(x)
            return I_hat, M_hat

        if self.mode == 'splitnet':
            I_output, M_hat, _ = self.model(x)
            I_hat = I_output[0] * M_hat + x * (1 - M_hat)
            return I_hat, M_hat

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
