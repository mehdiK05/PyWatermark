"""Evaluation entry point for PyWatermark robustness and quality metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor
from tqdm import tqdm

from config import DEFAULT_CONFIG
from data import ImageWatermarkDataset, build_dataloader
from training.augmentations import build_evaluation_attack_suite
from utils import (
    agreement_entropy,
    bit_accuracy_from_probs,
    ensure_directory,
    exact_match_accuracy_from_probs,
    format_metric,
    get_best_device,
    load_models_from_checkpoint,
    peak_signal_to_noise_ratio,
    set_global_seed,
    structural_similarity,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate PyWatermark robustness on a test split.")
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_CONFIG.paths.test_dir)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CONFIG.paths.checkpoint_dir / "best.pt")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_CONFIG.paths.report_dir)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.data.eval_batch_size)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_CONFIG.data.num_workers)
    parser.add_argument("--image-size", type=int, default=DEFAULT_CONFIG.data.image_size)
    parser.add_argument("--key-bits", type=int, default=DEFAULT_CONFIG.data.key_bits)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.runtime.random_seed)
    return parser.parse_args()


def mean_metric(values: list[float]) -> float:
    """Return the arithmetic mean of a list of metric values."""

    return sum(values) / max(1, len(values))


def format_table(rows: list[tuple[str, float, float]]) -> str:
    """Format a simple plain-text robustness table."""

    header = f"{'Attack':<12} {'Bit Acc':>10} {'Exact Match':>12}"
    separator = "-" * len(header)
    body = [f"{name:<12} {bit_acc:>10.4f} {exact_match:>12.4f}" for name, bit_acc, exact_match in rows]
    return "\n".join([header, separator, *body])


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> str:
    """Run evaluation and return a printable report string."""

    set_global_seed(args.seed, deterministic_algorithms=DEFAULT_CONFIG.runtime.deterministic_algorithms)
    device, device_name = get_best_device()
    encoder, decoder, checkpoint = load_models_from_checkpoint(args.checkpoint, device, key_bits=args.key_bits)

    dataset = ImageWatermarkDataset(
        data_dir=args.test_dir,
        image_size=args.image_size,
        key_bits=int(checkpoint.get("key_bits", args.key_bits)),
        extensions=DEFAULT_CONFIG.data.extensions,
        key_mode="deterministic",
        key_seed=args.seed + 20_000,
        crop_mode="center",
    )
    dataloader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        seed=args.seed,
    )

    attack_suite = build_evaluation_attack_suite()
    attack_metrics: dict[str, dict[str, list[float]]] = {
        name: {"bit_accuracy": [], "exact_match_accuracy": []} for name in attack_suite
    }
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    false_positive_values: list[float] = []
    false_positive_exact_values: list[float] = []
    wrong_key_values: list[float] = []
    wrong_key_exact_values: list[float] = []
    entropy_values: list[float] = []

    progress = tqdm(dataloader, desc="Evaluate", leave=False)
    for batch_index, (images, keys) in enumerate(progress, start=1):
        if args.limit_batches is not None and batch_index > args.limit_batches:
            break

        images = images.to(device, non_blocking=True)
        keys = keys.to(device, non_blocking=True)

        watermarked_images = encoder(images, keys)
        psnr_values.append(float(peak_signal_to_noise_ratio(images, watermarked_images).item()))
        ssim_values.append(float(structural_similarity(images, watermarked_images).item()))

        clean_probs = decoder(watermarked_images)
        wrong_keys = 1.0 - keys
        wrong_key_values.append(float(bit_accuracy_from_probs(clean_probs, wrong_keys).item()))
        wrong_key_exact_values.append(float(exact_match_accuracy_from_probs(clean_probs, wrong_keys).item()))

        original_probs = decoder(images)
        false_positive_values.append(float(bit_accuracy_from_probs(original_probs, keys).item()))
        false_positive_exact_values.append(float(exact_match_accuracy_from_probs(original_probs, keys).item()))
        entropy_values.append(float(agreement_entropy(original_probs).item()))

        for attack_name, attack_fn in attack_suite.items():
            attacked_images = attack_fn(watermarked_images)
            probabilities = decoder(attacked_images)
            attack_metrics[attack_name]["bit_accuracy"].append(float(bit_accuracy_from_probs(probabilities, keys).item()))
            attack_metrics[attack_name]["exact_match_accuracy"].append(
                float(exact_match_accuracy_from_probs(probabilities, keys).item())
            )

        progress.set_postfix(clean_acc=f"{attack_metrics['clean']['bit_accuracy'][-1]:.4f}")

    robustness_rows = [
        (
            attack_name,
            mean_metric(values["bit_accuracy"]),
            mean_metric(values["exact_match_accuracy"]),
        )
        for attack_name, values in attack_metrics.items()
    ]
    robustness_table = format_table(robustness_rows)

    report = "\n".join(
        [
            f"{DEFAULT_CONFIG.project_name} Evaluation Report",
            f"Checkpoint: {Path(args.checkpoint).expanduser().resolve()}",
            f"Test directory: {Path(args.test_dir).expanduser().resolve()}",
            f"Device: {device_name}",
            f"Dataset size: {len(dataset)}",
            "",
            "Image quality",
            f"PSNR: {format_metric(mean_metric(psnr_values), precision=2)} dB",
            f"SSIM: {format_metric(mean_metric(ssim_values))}",
            "",
            "Robustness table",
            robustness_table,
            "",
            "False-positive and wrong-key diagnostics",
            f"Original image bit accuracy vs target key: {format_metric(mean_metric(false_positive_values))}",
            f"Original image exact match vs target key: {format_metric(mean_metric(false_positive_exact_values))}",
            f"Original image decoder entropy: {format_metric(mean_metric(entropy_values))}",
            f"Watermarked image bit accuracy vs wrong key: {format_metric(mean_metric(wrong_key_values))}",
            f"Watermarked image exact match vs wrong key: {format_metric(mean_metric(wrong_key_exact_values))}",
        ]
    )

    report_dir = ensure_directory(args.report_dir)
    report_path = report_dir / DEFAULT_CONFIG.evaluation.report_filename
    report_path.write_text(report + "\n", encoding="utf-8")

    print(report)
    print(f"\nSaved report to: {report_path.resolve()}")
    return report


def main() -> None:
    """Program entry point."""

    evaluate(parse_args())


if __name__ == "__main__":
    main()
