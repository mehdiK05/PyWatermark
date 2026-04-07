# PyWatermark

PyWatermark is an invisible image watermarking project built with PyTorch. This repository is currently set up for Phase 1 only: configuration, dataset loading, dataloader construction, device selection, and basic pipeline validation.

## Current Status

Only Phase 1 is implemented.

Implemented in this phase:

- Project structure and placeholders for later phases
- Centralized configuration defaults in `config.py`
- Recursive image dataset with random crops and random binary keys
- Reusable dataloader builder
- Device selection helper for CUDA, MPS, or CPU
- Lightweight validation script in `training/train.py`

Not implemented yet:

- Encoder and decoder models
- Loss functions
- Robustness augmentations
- Training loop
- Evaluation pipeline
- CLI watermarking operations
- Gradio demo

## Installation

1. Create and activate a Python 3.10+ virtual environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Expected Image Data Layout

The Phase 1 dataset loader scans images recursively, so it supports nested image folders such as extracted DIV2K-style or CIFAR-style directory layouts.

Example:

```text
datasets/
├── train/
│   ├── image_0001.png
│   ├── image_0002.jpg
│   └── subset_a/
│       ├── sample_01.png
│       └── sample_02.webp
└── validation/
    └── preview.bmp
```

Supported extensions:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.webp`

## Dataset Validation

Use the Phase 1 validation entry point to verify that images are discovered, cropped, batched, and paired with fresh random binary keys:

```bash
python -m training.train --data-dir datasets/train --batch-size 8 --num-workers 0 --key-bits 48 --num-batches-preview 2
```

The script prints:

- Dataset size
- Selected device
- Batch image shape
- Batch key shape
- Image value range
- Unique key values

It does not start training yet.

## Notes

- Images smaller than `128x128` are upscaled before cropping.
- Image tensors are returned in `[0, 1]` with shape `(3, 128, 128)`.
- Each sample gets a newly generated random binary key on every `__getitem__` call.
- The repository is designed to run on CPU without code changes.

## Upcoming Phases

Future phases will add the encoder/decoder, losses, augmentations, training loop, evaluation workflow, CLI operations, and Gradio demo.
