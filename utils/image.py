"""Image loading, saving, and visualization helpers for PyWatermark."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F


def pil_image_to_tensor(image: Image.Image) -> Tensor:
    """Convert a PIL RGB image to a float tensor in ``[0, 1]``."""

    rgb_image = image.convert("RGB")
    array = np.asarray(rgb_image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def tensor_to_pil_image(image: Tensor) -> Image.Image:
    """Convert a tensor image in ``[0, 1]`` to a PIL image."""

    image_3d = image.detach().cpu()
    if image_3d.ndim == 4:
        if image_3d.shape[0] != 1:
            raise ValueError("tensor_to_pil_image expects a single image tensor.")
        image_3d = image_3d[0]
    if image_3d.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dimensions, received {tuple(image_3d.shape)}.")

    image_uint8 = image_3d.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
    return Image.fromarray(image_uint8.permute(1, 2, 0).numpy(), mode="RGB")


def load_image_tensor(path: str | Path, image_size: int | None = None) -> Tensor:
    """Load an image file into a 4D tensor of shape ``(1, C, H, W)``."""

    image_path = Path(path).expanduser()
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        if image_size is not None:
            rgb_image = rgb_image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        tensor = pil_image_to_tensor(rgb_image)
    return tensor.unsqueeze(0)


def save_image_tensor(image: Tensor, path: str | Path) -> Path:
    """Save a tensor image in ``[0, 1]`` to disk and return its path."""

    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_image = tensor_to_pil_image(image)
    pil_image.save(output_path)
    return output_path


def resize_image_tensor(image: Tensor, image_size: int) -> Tensor:
    """Resize a 4D tensor image batch to a square spatial size."""

    return F.interpolate(
        image,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )


def make_image_grid(images: Tensor, max_images: int = 4) -> Tensor:
    """Arrange a batch of images into a simple grid tensor."""

    if images.ndim != 4:
        raise ValueError(f"Expected image batch of shape (B, C, H, W), received {tuple(images.shape)}.")

    count = min(images.shape[0], max_images)
    images = images[:count].detach().cpu().clamp(0.0, 1.0)
    rows = int(math.ceil(math.sqrt(count)))
    cols = int(math.ceil(count / rows))
    height, width = images.shape[-2:]
    grid = torch.zeros(images.shape[1], rows * height, cols * width, dtype=images.dtype)

    for index in range(count):
        row = index // cols
        col = index % cols
        grid[:, row * height : (row + 1) * height, col * width : (col + 1) * width] = images[index]

    return grid
