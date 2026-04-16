# PyWatermark

PyWatermark is a from-scratch invisible image watermarking system in PyTorch, built as a compact research-style project inspired by SynthID-style training loops: an encoder hides a binary key inside an image, a decoder recovers the key after differentiable distortions, and the system is trained end-to-end for a quality/robustness tradeoff.

The repository is designed to be:

- runnable on CPU for debugging
- GPU-friendly for training
- Colab-friendly for longer runs
- clean enough to serve as a portfolio project rather than a one-off experiment

## Features

- Centralized configuration in [`config.py`](config.py)
- Recursive image datasets with `train/`, `val/`, and `test/` splits
- Fresh random binary keys during training
- Deterministic keys and center crops for validation and test
- UNet-lite encoder with bottleneck key injection and bounded residual output
- Lightweight ResNet-style decoder with 4 residual blocks
- Differentiable robustness pipeline:
  Gaussian blur, crop/resize, color jitter, Gaussian noise, and JPEG-style compression approximation
- End-to-end training with:
  Adam, gradient clipping, `ReduceLROnPlateau`, checkpoints, resume, TensorBoard, and image previews
- OOM-aware batch-size fallback during training
- Evaluation script with robustness table and false-positive / wrong-key diagnostics
- CLI for embedding and detecting watermarks
- Gradio demo with embed and detect tabs
- Minimal Colab notebooks for training and evaluation

## Project Layout

```text
PyWatermark/
├── cli.py
├── config.py
├── data/
├── demo/
├── evaluation/
├── models/
├── notebooks/
├── training/
├── utils/
├── requirements.txt
└── README.md
```

## Dataset Layout

Expected directory structure:

```text
datasets/
├── train/
│   ├── image_0001.png
│   └── nested_folder/sample.jpg
├── val/
│   └── image_0100.png
└── test/
    └── image_0200.png
```

Supported extensions:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.webp`

Notes:

- Images are converted to RGB and resized only when needed to allow valid cropping.
- Training uses random square crops.
- Validation and test use deterministic center crops for stable metrics.
- Image tensors are normalized to `[0, 1]`.

## Prepare COCO by Code

You can build the dataset directly in Colab or on any machine without manually downloading on your PC first.

Recommended lightweight Colab setup:

- download only `val2017`
- split it into `train`, `val`, and `test`
- store everything in Google Drive

Mount Drive in Colab:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then run the prep utility from the project root:

```bash
python -m data.prepare_coco \
  --download-split val2017 \
  --raw-dir /content/drive/MyDrive/PyWatermark/raw \
  --output-root /content/drive/MyDrive/PyWatermark/datasets \
  --train-count 4000 \
  --val-count 500 \
  --test-count 500 \
  --force
```

This downloads the official COCO `val2017.zip`, extracts it, and creates:

```text
/content/drive/MyDrive/PyWatermark/datasets/
├── train/
├── val/
└── test/
```

Useful variants:

- Limit the pool before splitting:

```bash
python -m data.prepare_coco \
  --download-split val2017 \
  --raw-dir /content/drive/MyDrive/PyWatermark/raw \
  --output-root /content/drive/MyDrive/PyWatermark/datasets \
  --max-images 2000 \
  --train-count 1500 \
  --val-count 250 \
  --test-count 250 \
  --force
```

- Split an already extracted image folder instead of downloading:

```bash
python -m data.prepare_coco \
  --source-dir /content/drive/MyDrive/PyWatermark/raw/val2017 \
  --output-root /content/drive/MyDrive/PyWatermark/datasets \
  --train-count 4000 \
  --val-count 500 \
  --test-count 500 \
  --force
```

When `--source-dir` is used, files are copied by default so the original image folder stays intact. When the utility downloads COCO itself, files are moved into `datasets/` by default to save space.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS activation:

```bash
source .venv/bin/activate
```

## Training

Local training:

```bash
python -m training.train \
  --train-dir datasets/train \
  --val-dir datasets/val \
  --test-dir datasets/test \
  --checkpoint-dir artifacts/checkpoints \
  --log-dir artifacts/logs \
  --epochs 20 \
  --train-batch-size 8 \
  --eval-batch-size 8 \
  --num-workers 0
```

Resume from the latest checkpoint:

```bash
python -m training.train \
  --train-dir datasets/train \
  --val-dir datasets/val \
  --test-dir datasets/test \
  --checkpoint-dir artifacts/checkpoints \
  --log-dir artifacts/logs \
  --auto-resume
```

CPU smoke test:

```bash
python -m training.train \
  --train-dir datasets/train \
  --val-dir datasets/val \
  --test-dir datasets/test \
  --num-workers 0 \
  --train-batch-size 2 \
  --eval-batch-size 2 \
  --run-smoke-test
```

TensorBoard:

```bash
tensorboard --logdir artifacts/logs
```

## Evaluation

```bash
python -m evaluation.evaluate \
  --test-dir datasets/test \
  --checkpoint artifacts/checkpoints/best.pt \
  --report-dir artifacts/reports \
  --batch-size 8 \
  --num-workers 0
```

This prints a report and saves `artifacts/reports/evaluation_report.txt`.

The evaluator reports:

- bit accuracy
- exact match accuracy
- PSNR
- SSIM
- robustness on clean / JPEG / blur / crop / noise / brightness
- false-positive behavior on non-watermarked images
- wrong-key behavior on watermarked images

## CLI

Embed a watermark:

```bash
python cli.py embed \
  --checkpoint artifacts/checkpoints/best.pt \
  --input path/to/input.png \
  --output artifacts/outputs/watermarked.png \
  --key 010101010101010101010101010101010101010101010101
```

Embed with a generated random key:

```bash
python cli.py embed \
  --checkpoint artifacts/checkpoints/best.pt \
  --input path/to/input.png \
  --output artifacts/outputs/watermarked.png
```

Detect from a watermarked image:

```bash
python cli.py detect \
  --checkpoint artifacts/checkpoints/best.pt \
  --input artifacts/outputs/watermarked.png
```

Detect against an expected key:

```bash
python cli.py detect \
  --checkpoint artifacts/checkpoints/best.pt \
  --input artifacts/outputs/watermarked.png \
  --key 010101010101010101010101010101010101010101010101
```

## Gradio Demo

```bash
python -m demo.app --checkpoint artifacts/checkpoints/best.pt
```

Default behavior launches with `share=True` for Colab-friendly usage. Disable sharing locally with:

```bash
python -m demo.app --checkpoint artifacts/checkpoints/best.pt --no-share
```

## Colab Workflow

Minimal notebooks are provided in:

- [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb)
- [`notebooks/evaluate_colab.ipynb`](notebooks/evaluate_colab.ipynb)

The training notebook now includes the dataset prep step, so you can mount Drive, download a small COCO subset by code, split it into `train/val/test`, and start training in one Colab workflow.

Typical Colab storage pattern:

- dataset in `/content/drive/MyDrive/PyWatermark/datasets/...`
- checkpoints in `/content/drive/MyDrive/PyWatermark/artifacts/checkpoints`
- logs in `/content/drive/MyDrive/PyWatermark/artifacts/logs`
- reports in `/content/drive/MyDrive/PyWatermark/artifacts/reports`

Example Colab training command:

```bash
python -m training.train \
  --train-dir /content/drive/MyDrive/PyWatermark/datasets/train \
  --val-dir /content/drive/MyDrive/PyWatermark/datasets/val \
  --test-dir /content/drive/MyDrive/PyWatermark/datasets/test \
  --checkpoint-dir /content/drive/MyDrive/PyWatermark/artifacts/checkpoints \
  --log-dir /content/drive/MyDrive/PyWatermark/artifacts/logs \
  --epochs 50 \
  --train-batch-size 16 \
  --eval-batch-size 16 \
  --num-workers 2 \
  --auto-resume
```

Example Colab evaluation command:

```bash
python -m evaluation.evaluate \
  --test-dir /content/drive/MyDrive/PyWatermark/datasets/test \
  --checkpoint /content/drive/MyDrive/PyWatermark/artifacts/checkpoints/best.pt \
  --report-dir /content/drive/MyDrive/PyWatermark/artifacts/reports \
  --batch-size 16 \
  --num-workers 2
```

## Architecture Summary

Encoder:

- 3 down blocks
- key injection at the bottleneck by spatial broadcast + channel concatenation
- 3 up blocks with skip connections
- residual head with `tanh` bounded by `alpha`
- final watermarked image is `clamp(original + residual, 0, 1)`

Decoder:

- convolutional stem
- 4 residual blocks
- downsampling after blocks 2 and 4
- global average pooling
- fully connected projection to `N` watermark bits

Loss:

- invisibility loss = `(1 - SSIM) + L2`, with configurable weighting
- detection loss = BCE over watermark bits
- total loss = weighted sum of invisibility and detection terms

## Key Assumptions

- The model operates on square images of a fixed training size set in `config.py`.
- Validation and test use deterministic center crops for stable metrics.
- The implemented JPEG attack is a differentiable JPEG-style approximation based on resize + quantization, which keeps gradients flowing without relying on an external pretrained watermarking package.
- Wrong-key evaluation compares decoded watermark bits against the complemented key vector.
- Detection is decoder-based: the decoder predicts watermark bits directly, and optional key comparison is done outside the network.
- If TensorBoard is unavailable, training still runs with logging disabled rather than crashing.

## Known Limitations

- This is a compact baseline, not a reproduction of SynthID internals.
- CLI and demo inference resize inputs to the model training resolution; they do not preserve arbitrary original resolutions.
- Robustness quality depends heavily on actual training time and dataset diversity.
- The OOM fallback reduces batch size and retries the epoch, but it does not dynamically rebalance other hyperparameters such as learning rate.
- If `kornia` is not installed, color jitter falls back to a manual differentiable implementation.

## What To Run on GPU vs CPU

GPU recommended:

- full training
- longer validation/evaluation runs on larger datasets
- demo hosting with repeated inference under load

CPU is fine for:

- import checks
- forward-pass debugging
- one-batch smoke tests
- CLI sanity checks
- very small evaluation subsets
