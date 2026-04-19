# Mimicry

Mimicry is a compact, from-scratch invisible image watermarking project in PyTorch. The goal is not to reproduce Google SynthID exactly, but to build an honest, inspectable approximation of the same broad idea: an encoder hides a binary key inside an image, a decoder attempts to recover that key after common distortions, and the whole system is trained end-to-end around the tradeoff between robustness and image fidelity.

The repo now reflects the final project state:

- final robust checkpoint stored in `artifacts/checkpoints/best.pt`
- final evaluation report in `artifacts/report/evaluation_report.txt`
- final training plots in `artifacts/results/`

## Final Result

Final test evaluation for the robust 16-bit checkpoint:

- average attacked bit accuracy: `0.6465`
- clean bit accuracy: `0.6528`
- JPEG bit accuracy: `0.6536`
- blur bit accuracy: `0.6472`
- crop bit accuracy: `0.6300`
- noise bit accuracy: `0.6503`
- brightness bit accuracy: `0.6450`
- PSNR: `23.52 dB`
- SSIM: `0.8642`

This is a robustness-first result. The model is clearly above random decoding, but it does not achieve perfect key recovery and still shows the expected robustness-quality tradeoff.

## Repository Layout

```text
Mimicry/
├── artifacts/
│   ├── checkpoints/
│   │   └── best.pt
│   ├── report/
│   │   └── evaluation_report.txt
│   └── results/
│       ├── bit_accuracy_curve.png
│       ├── loss_curve.png
│       ├── psnr_curve.png
│       ├── ssim_curve.png
│       └── training_summary.png
├── data/
├── demo/
├── evaluation/
├── models/
├── notebooks/
├── training/
├── utils/
├── cli.py
├── config.py
└── requirements.txt
```

## Core Components

- `models/encoder.py`
  UNet-lite encoder that injects a binary key at the bottleneck and outputs a bounded residual.
- `models/decoder.py`
  Lightweight ResNet-style decoder that predicts watermark bits from an image.
- `training/augmentations.py`
  Differentiable attacks used during robust training, including JPEG-style compression, blur, crop-resize, color jitter, and Gaussian noise.
- `training/train.py`
  Main training script with checkpointing, resume support, CSV history logging, and configurable loss weights.
- `evaluation/evaluate.py`
  Test-set evaluator that reports image quality, attack robustness, and false-positive / wrong-key diagnostics.
- `evaluation/plot_training_curves.py`
  Utility that generates report-ready PNG plots from `metrics_history.csv`.
- `demo/app.py`
  Gradio demo for interactive embedding and detection.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

## Run the Demo

The local final checkpoint is already placed where the demo expects it.

```bash
python -m demo.app --no-share
```

If you want to be explicit:

```bash
python -m demo.app --checkpoint artifacts/checkpoints/best.pt --no-share
```

Then open:

- `http://127.0.0.1:7860`
- or `http://localhost:7860`

If you are on a remote VS Code / Codespaces environment, forward port `7860` and open the forwarded URL.

## CLI Usage

Embed a watermark:

```bash
python cli.py embed ^
  --checkpoint artifacts/checkpoints/best.pt ^
  --input path\to\input.png ^
  --output artifacts\outputs\watermarked.png
```

Detect a watermark:

```bash
python cli.py detect ^
  --checkpoint artifacts/checkpoints/best.pt ^
  --input artifacts\outputs\watermarked.png
```

Important:

- the checkpoint currently uses a `16`-bit payload
- the CLI and demo resize inputs to the model resolution (`128x128`)
- the expected key for accuracy checks must be the original embedded key, not the decoder output

## Training Summary

The final reported model came from a staged robust training process:

1. a clean bootstrap phase without attacks
2. a robust training phase with differentiable distortions
3. a stronger robustness-focused continuation phase with higher detection pressure and reduced invisibility emphasis

This change in objective explains the visible jump in training loss around the later epochs in the saved plots. The jump is not a plotting error or collapse; it marks a regime change where the model began favoring watermark recoverability more aggressively, with a corresponding drop in perceptual quality.

## Evaluation Summary

The evaluator reports:

- bit accuracy
- exact match accuracy
- PSNR
- SSIM
- robustness under clean, JPEG, blur, crop, noise, and brightness shifts
- false-positive behavior on original images
- wrong-key behavior on watermarked images

`Exact match` is near zero in the final model because full `16/16` bit recovery remains much harder than partial recovery. The model often recovers a majority of bits correctly without achieving a perfect full-key decode.

## Notebooks

Two Colab notebooks are included:

- `notebooks/train_colab.ipynb`
- `notebooks/evaluate_colab.ipynb`

They are now written as clean experiment notebooks rather than debugging logs:

- training notebook: dataset prep, environment setup, clean bootstrap, robust training, and plot export
- evaluation notebook: checkpoint evaluation, report display, compact metric summary, and plot export

## Known Limitations

- Mimicry is inspired by SynthID, not a faithful reproduction of Google’s internal system.
- The final robust model is moderate rather than state-of-the-art.
- Crop robustness remains weaker than other tested distortions.
- The final checkpoint preserves some robustness, but exact full-key recovery is still rare.
- Evaluation and demo inference operate on resized square images rather than arbitrary original resolutions.

## Next Research Directions

- select `best.pt` using robust validation rather than clean validation
- expand decoder capacity further
- explore smaller payloads such as `8` bits for stronger clean recovery
- improve crop robustness specifically
- train on larger and more diverse datasets
- add stronger perceptual losses and better scheduling curricula

## License / Attribution

This project is an independent educational and research-style implementation. It is best described as a SynthID-inspired mimic rather than an official reproduction.
