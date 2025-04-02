import torch
from torch import Tensor


class ImageList:
    def __init__(self, images_tensor: Tensor) -> None:
        self.tensors = images_tensor

        batch_size = images_tensor.shape[0]
        image_sizes = images_tensor.shape[-2:]

        self.image_sizes = [tuple(image_sizes) for _ in range(batch_size)]

    def to(self, device: torch.device) -> "ImageList":
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)
