"""Export training curves from a PyWatermark metrics CSV history file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for plotting training curves."""

    parser = argparse.ArgumentParser(description="Plot PyWatermark training curves from metrics_history.csv.")
    parser.add_argument("--history", type=Path, required=True, help="Path to metrics_history.csv.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store PNG figures.")
    parser.add_argument("--title-prefix", type=str, default="PyWatermark", help="Prefix used in plot titles.")
    return parser.parse_args()


def read_history(path: Path) -> dict[str, list[float]]:
    """Read a metrics history CSV file into column arrays."""

    if not path.exists():
        raise FileNotFoundError(f"History file not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found in history file: {path}")
        data = {field: [] for field in reader.fieldnames}
        for row in reader:
            for field in reader.fieldnames:
                data[field].append(float(row[field]))
    if not data["epoch"]:
        raise ValueError(f"No rows found in history file: {path}")
    return data


def save_line_plot(
    epochs: list[float],
    train_values: list[float],
    val_values: list[float],
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Save a simple train-vs-val line plot."""

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(epochs, train_values, label="Train", linewidth=2)
    axis.plot(epochs, val_values, label="Validation", linewidth=2)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_summary_grid(history: dict[str, list[float]], title_prefix: str, output_path: Path) -> None:
    """Save a 2x2 summary grid covering the main report metrics."""

    import matplotlib.pyplot as plt

    epochs = history["epoch"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_specs = [
        ("Loss", "train_total_loss", "val_total_loss"),
        ("Bit Accuracy", "train_bit_accuracy", "val_bit_accuracy"),
        ("PSNR (dB)", "train_psnr", "val_psnr"),
        ("SSIM", "train_ssim", "val_ssim"),
    ]

    for axis, (ylabel, train_key, val_key) in zip(axes.flatten(), plot_specs):
        axis.plot(epochs, history[train_key], label="Train", linewidth=2)
        axis.plot(epochs, history[val_key], label="Validation", linewidth=2)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.set_title(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.suptitle(f"{title_prefix} Training Summary", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Program entry point."""

    args = parse_args()
    history = read_history(args.history.expanduser())
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = history["epoch"]

    save_line_plot(
        epochs,
        history["train_total_loss"],
        history["val_total_loss"],
        ylabel="Loss",
        title=f"{args.title_prefix} Loss Curve",
        output_path=output_dir / "loss_curve.png",
    )
    save_line_plot(
        epochs,
        history["train_bit_accuracy"],
        history["val_bit_accuracy"],
        ylabel="Bit Accuracy",
        title=f"{args.title_prefix} Bit Accuracy Curve",
        output_path=output_dir / "bit_accuracy_curve.png",
    )
    save_line_plot(
        epochs,
        history["train_psnr"],
        history["val_psnr"],
        ylabel="PSNR (dB)",
        title=f"{args.title_prefix} PSNR Curve",
        output_path=output_dir / "psnr_curve.png",
    )
    save_line_plot(
        epochs,
        history["train_ssim"],
        history["val_ssim"],
        ylabel="SSIM",
        title=f"{args.title_prefix} SSIM Curve",
        output_path=output_dir / "ssim_curve.png",
    )
    save_summary_grid(
        history=history,
        title_prefix=args.title_prefix,
        output_path=output_dir / "training_summary.png",
    )
    print(f"Saved plots to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
