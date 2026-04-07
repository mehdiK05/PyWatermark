"""Dataset and dataloader helpers for Phase 1 of PyWatermark."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as transforms_f

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_KEY_BITS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SHUFFLE,
    TRAIN_EXTENSIONS,
)


def collect_image_paths(
    data_dir: str | Path,
    extensions: Sequence[str] | None = None,
) -> list[Path]:
    """Collect image file paths recursively from a directory."""

    root_dir = Path(data_dir).expanduser()
    if not root_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root_dir}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Data directory is not a directory: {root_dir}")

    normalized_extensions = {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in (extensions or TRAIN_EXTENSIONS)
    }

    image_paths = [
        path
        for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized_extensions
    ]
    image_paths.sort()
    return image_paths


class ImageWatermarkDataset(Dataset[tuple[Tensor, Tensor]]):
    """Load image crops and freshly sampled binary watermark keys."""

    def __init__(
        self,
        data_dir: str | Path,
        image_size: int = DEFAULT_IMAGE_SIZE,
        key_bits: int = DEFAULT_KEY_BITS,
        extensions: Sequence[str] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser()
        self.image_size = image_size
        self.key_bits = key_bits
        self.extensions = tuple(extensions or TRAIN_EXTENSIONS)
        self.image_paths = collect_image_paths(self.data_dir, self.extensions)

    def __len__(self) -> int:
        """Return the number of discovered images."""

        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Return a random crop and a newly sampled binary key."""

        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            processed_image = self._prepare_image(rgb_image)

        image_tensor = transforms_f.to_tensor(processed_image)
        key_tensor = self._sample_key()
        return image_tensor, key_tensor

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        """Resize images when needed and extract a random square crop."""

        prepared_image = self._resize_if_needed(image)
        return self._random_crop(prepared_image)

    def _resize_if_needed(self, image: Image.Image) -> Image.Image:
        """Upscale small images so random cropping remains valid."""

        width, height = image.size
        if width >= self.image_size and height >= self.image_size:
            return image

        scale = max(self.image_size / width, self.image_size / height)
        resized_width = max(self.image_size, int(round(width * scale)))
        resized_height = max(self.image_size, int(round(height * scale)))
        return image.resize(
            (resized_width, resized_height),
            resample=Image.Resampling.BICUBIC,
        )

    def _random_crop(self, image: Image.Image) -> Image.Image:
        """Extract a random square crop from the image."""

        width, height = image.size
        max_top = height - self.image_size
        max_left = width - self.image_size
        top = 0 if max_top == 0 else int(torch.randint(0, max_top + 1, (1,)).item())
        left = 0 if max_left == 0 else int(torch.randint(0, max_left + 1, (1,)).item())
        return transforms_f.crop(image, top, left, self.image_size, self.image_size)

    def _sample_key(self) -> Tensor:
        """Sample a new binary key vector in float32 format."""

        return torch.randint(0, 2, (self.key_bits,), dtype=torch.int64).to(torch.float32)


def build_dataloader(
    dataset: Dataset[tuple[Tensor, Tensor]],
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = DEFAULT_SHUFFLE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    pin_memory: bool | None = None,
    drop_last: bool = False,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Build a reusable dataloader for the watermarking pipeline."""

    effective_pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=effective_pin_memory,
        drop_last=drop_last,
    )
