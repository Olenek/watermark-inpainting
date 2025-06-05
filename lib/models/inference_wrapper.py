from typing import cast

import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms

from lib.models.rirci import RIRCIModel
from lib.models.splitnet_interface import SplitNet


class Denormalize(object):
    """Denormalize a tensor image with mean/std or scale (for [-1, 1])."""

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        denorm_tensor = tensor.clone()

        if self.mean is not None and self.std is not None:
            dtype = denorm_tensor.dtype
            mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device).view(-1, 1, 1)
            std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device).view(-1, 1, 1)
            denorm_tensor.mul_(std).add_(mean)

        return torch.clamp(denorm_tensor, 0, 1)

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


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

    def enhance_image(self, device, image_path, norm=False) -> Image:
        input_image = Image.open(image_path).convert('RGB')

        preprocess, postprocess = self._make_pre_post_processing(norm, input_image.size)

        # Apply transformations and add batch dimension
        input_tensor = preprocess(input_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor, _ = self.forward(input_tensor)

        output_tensor = postprocess(output_tensor).squeeze(0)
        output_image = transforms.ToPILImage()(output_tensor.cpu())

        return output_image

    @staticmethod
    def _make_pre_post_processing(use_norm: bool, original_shape: tuple[int, int]):
        if use_norm:
            preprocess = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

            post_process = transforms.Compose([
                Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                # transforms.Resize((original_shape[1], original_shape[0])),
            ])
        else:
            preprocess = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])

            post_process = transforms.Compose([
                Denormalize(),  # Just clamp
                # transforms.Resize((original_shape[1], original_shape[0])),
            ])

        return preprocess, post_process
