"""Phase 1 dataset and dataloader validation entry point for PyWatermark."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from config import (
    DEFAULT_CONFIG,
    DEFAULT_DATA_DIR,
    DEFAULT_KEY_BITS,
    DEFAULT_NUM_BATCHES_PREVIEW,
    DEFAULT_NUM_WORKERS,
    RANDOM_SEED,
    TRAIN_EXTENSIONS,
)
from data import ImageWatermarkDataset, build_dataloader
from utils import get_best_device


def parse_args() -> argparse.Namespace:
    """Parse Phase 1 validation arguments."""

    parser = argparse.ArgumentParser(
        description="Validate the Phase 1 dataset and dataloader pipeline.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing training images.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_CONFIG.data.batch_size,
        help="Batch size used for preview loading.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--key-bits",
        type=int,
        default=DEFAULT_KEY_BITS,
        help="Number of random binary watermark bits per sample.",
    )
    parser.add_argument(
        "--num-batches-preview",
        type=int,
        default=DEFAULT_NUM_BATCHES_PREVIEW,
        help="How many batches to preview before exiting.",
    )
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    """Seed Python and PyTorch RNGs for reproducible validation runs."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preview_batch(batch_index: int, images: torch.Tensor, keys: torch.Tensor) -> None:
    """Print lightweight batch diagnostics."""

    unique_key_values = torch.unique(keys).cpu().tolist()
    print(f"Batch {batch_index}:")
    print(f"  image shape: {tuple(images.shape)}")
    print(f"  key shape: {tuple(keys.shape)}")
    print(f"  image min/max: {images.min().item():.4f} / {images.max().item():.4f}")
    print(f"  unique key values: {unique_key_values}")


def main() -> None:
    """Run the Phase 1 dataset pipeline sanity check."""

    args = parse_args()
    set_random_seed(RANDOM_SEED)

    device, device_name = get_best_device()
    dataset = ImageWatermarkDataset(
        data_dir=args.data_dir,
        image_size=DEFAULT_CONFIG.data.image_size,
        key_bits=args.key_bits,
        extensions=TRAIN_EXTENSIONS,
    )

    if len(dataset) == 0:
        raise FileNotFoundError(
            f"No images were found in '{args.data_dir}'. "
            f"Supported extensions: {', '.join(TRAIN_EXTENSIONS)}"
        )

    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=DEFAULT_CONFIG.data.shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    print(f"Project: {DEFAULT_CONFIG.project_name}")
    print(f"Dataset directory: {Path(args.data_dir).expanduser().resolve()}")
    print(f"Dataset size: {len(dataset)} images")
    print(f"Selected device: {device_name}")

    for batch_index, (images, keys) in enumerate(dataloader, start=1):
        preview_batch(batch_index, images, keys)
        if batch_index >= args.num_batches_preview:
            break

    print("Phase 1 dataset validation completed successfully.")


if __name__ == "__main__":
    main()
