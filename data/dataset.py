"""Dataset and dataloader helpers for the full PyWatermark pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from config import DEFAULT_CONFIG, TRAIN_EXTENSIONS
from utils.seed import seed_worker

KeyMode = Literal["random", "deterministic"]
CropMode = Literal["random", "center"]


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


def _pil_to_tensor(image: Image.Image) -> Tensor:
    """Convert a PIL image into a normalized float tensor."""

    array = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32)) / 255.0
    return array.permute(2, 0, 1).contiguous()


class ImageWatermarkDataset(Dataset[tuple[Tensor, Tensor]]):
    """Load square image crops and paired watermark key tensors."""

    def __init__(
        self,
        data_dir: str | Path,
        image_size: int = DEFAULT_CONFIG.data.image_size,
        key_bits: int = DEFAULT_CONFIG.data.key_bits,
        extensions: Sequence[str] | None = None,
        key_mode: KeyMode = "random",
        key_seed: int = DEFAULT_CONFIG.runtime.random_seed,
        crop_mode: CropMode = "random",
    ) -> None:
        self.data_dir = Path(data_dir).expanduser()
        self.image_size = image_size
        self.key_bits = key_bits
        self.extensions = tuple(extensions or TRAIN_EXTENSIONS)
        self.key_mode = key_mode
        self.key_seed = key_seed
        self.crop_mode = crop_mode
        self.image_paths = collect_image_paths(self.data_dir, self.extensions)

        if self.key_bits <= 0:
            raise ValueError("key_bits must be positive.")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")

    def __len__(self) -> int:
        """Return the number of discovered images."""

        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Return a normalized image crop and a watermark key tensor."""

        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image_tensor = self._prepare_image(image.convert("RGB"))
        key_tensor = self._sample_key(index)
        return image_tensor, key_tensor

    def _prepare_image(self, image: Image.Image) -> Tensor:
        """Resize an image if needed and return a square crop tensor."""

        tensor = _pil_to_tensor(image).unsqueeze(0)
        tensor = self._resize_if_needed(tensor)
        tensor = self._crop_tensor(tensor)
        return tensor.squeeze(0).clamp(0.0, 1.0)

    def _resize_if_needed(self, image: Tensor) -> Tensor:
        """Upscale small images to ensure valid cropping."""

        height, width = image.shape[-2:]
        if height >= self.image_size and width >= self.image_size:
            return image

        scale = max(self.image_size / height, self.image_size / width)
        resized_height = max(self.image_size, int(round(height * scale)))
        resized_width = max(self.image_size, int(round(width * scale)))
        return F.interpolate(
            image,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )

    def _crop_tensor(self, image: Tensor) -> Tensor:
        """Return a random or centered square crop."""

        height, width = image.shape[-2:]
        max_top = height - self.image_size
        max_left = width - self.image_size

        if self.crop_mode == "center":
            top = max_top // 2
            left = max_left // 2
        else:
            top = 0 if max_top == 0 else int(torch.randint(0, max_top + 1, (1,)).item())
            left = 0 if max_left == 0 else int(torch.randint(0, max_left + 1, (1,)).item())

        return image[:, :, top : top + self.image_size, left : left + self.image_size]

    def _sample_key(self, index: int) -> Tensor:
        """Return either a fresh random key or a deterministic per-index key."""

        if self.key_mode == "deterministic":
            generator = torch.Generator()
            generator.manual_seed(self.key_seed + index)
            key = torch.randint(0, 2, (self.key_bits,), generator=generator, dtype=torch.int64)
        else:
            key = torch.randint(0, 2, (self.key_bits,), dtype=torch.int64)
        return key.to(torch.float32)


def build_dataloader(
    dataset: Dataset[tuple[Tensor, Tensor]],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool | None = None,
    drop_last: bool = False,
    seed: int | None = None,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Build a reproducible dataloader for the watermarking pipeline."""

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    effective_pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
    persistent_workers = num_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=effective_pin_memory,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=persistent_workers,
    )


def build_split_datasets(
    train_dir: str | Path,
    val_dir: str | Path,
    test_dir: str | Path | None = None,
    image_size: int = DEFAULT_CONFIG.data.image_size,
    key_bits: int = DEFAULT_CONFIG.data.key_bits,
    eval_key_seed: int = DEFAULT_CONFIG.data.eval_key_seed,
    extensions: Sequence[str] | None = None,
) -> dict[str, ImageWatermarkDataset]:
    """Construct train, validation, and test datasets with split-appropriate settings."""

    datasets: dict[str, ImageWatermarkDataset] = {
        "train": ImageWatermarkDataset(
            data_dir=train_dir,
            image_size=image_size,
            key_bits=key_bits,
            extensions=extensions,
            key_mode="random",
            crop_mode="random",
        ),
        "val": ImageWatermarkDataset(
            data_dir=val_dir,
            image_size=image_size,
            key_bits=key_bits,
            extensions=extensions,
            key_mode="deterministic",
            key_seed=eval_key_seed,
            crop_mode="center",
        ),
    }
    if test_dir is not None and Path(test_dir).expanduser().exists():
        datasets["test"] = ImageWatermarkDataset(
            data_dir=test_dir,
            image_size=image_size,
            key_bits=key_bits,
            extensions=extensions,
            key_mode="deterministic",
            key_seed=eval_key_seed + 10_000,
            crop_mode="center",
        )
    return datasets
