"""Training entry point for the PyWatermark watermarking system."""

from __future__ import annotations

import argparse
import csv
import time
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from config import DEFAULT_CONFIG
from data import build_dataloader, build_split_datasets
from models.decoder import WatermarkDecoder
from models.encoder import WatermarkEncoder
from training.augmentations import RandomAugmentationPipeline
from training.losses import WatermarkLoss
from utils import (
    bit_accuracy_from_probs,
    ensure_directory,
    exact_match_accuracy_from_probs,
    find_latest_checkpoint,
    get_best_device,
    load_checkpoint,
    make_image_grid,
    peak_signal_to_noise_ratio,
    save_checkpoint,
    set_global_seed,
    structural_similarity,
)


class NullSummaryWriter:
    """Fallback writer used when TensorBoard is unavailable."""

    def add_scalar(self, *_args: Any, **_kwargs: Any) -> None:
        """Ignore scalar logging when TensorBoard is unavailable."""

    def add_image(self, *_args: Any, **_kwargs: Any) -> None:
        """Ignore image logging when TensorBoard is unavailable."""

    def close(self) -> None:
        """Close the no-op writer."""


def append_epoch_history(
    log_dir: Path,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    learning_rate: float,
) -> Path:
    """Append a row of epoch metrics to a CSV history file."""

    ensure_directory(log_dir)
    history_path = log_dir / "metrics_history.csv"
    fieldnames = [
        "epoch",
        "learning_rate",
        "train_total_loss",
        "train_invisibility_loss",
        "train_detection_loss",
        "train_bit_accuracy",
        "train_exact_match_accuracy",
        "train_psnr",
        "train_ssim",
        "val_total_loss",
        "val_invisibility_loss",
        "val_detection_loss",
        "val_bit_accuracy",
        "val_exact_match_accuracy",
        "val_psnr",
        "val_ssim",
    ]
    row = {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "train_total_loss": train_metrics["total_loss"],
        "train_invisibility_loss": train_metrics["invisibility_loss"],
        "train_detection_loss": train_metrics["detection_loss"],
        "train_bit_accuracy": train_metrics["bit_accuracy"],
        "train_exact_match_accuracy": train_metrics["exact_match_accuracy"],
        "train_psnr": train_metrics["psnr"],
        "train_ssim": train_metrics["ssim"],
        "val_total_loss": val_metrics["total_loss"],
        "val_invisibility_loss": val_metrics["invisibility_loss"],
        "val_detection_loss": val_metrics["detection_loss"],
        "val_bit_accuracy": val_metrics["bit_accuracy"],
        "val_exact_match_accuracy": val_metrics["exact_match_accuracy"],
        "val_psnr": val_metrics["psnr"],
        "val_ssim": val_metrics["ssim"],
    }

    write_header = not history_path.exists()
    with history_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return history_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""

    parser = argparse.ArgumentParser(description="Train the PyWatermark encoder/decoder pair.")
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_CONFIG.paths.train_dir)
    parser.add_argument("--val-dir", type=Path, default=DEFAULT_CONFIG.paths.val_dir)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_CONFIG.paths.test_dir)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CONFIG.paths.checkpoint_dir)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_CONFIG.paths.log_dir)
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.training.epochs)
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_CONFIG.data.train_batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_CONFIG.data.eval_batch_size)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_CONFIG.data.num_workers)
    parser.add_argument("--image-size", type=int, default=DEFAULT_CONFIG.data.image_size)
    parser.add_argument("--key-bits", type=int, default=DEFAULT_CONFIG.data.key_bits)
    parser.add_argument("--encoder-alpha", type=float, default=DEFAULT_CONFIG.encoder.alpha)
    parser.add_argument("--encoder-base-channels", type=int, default=DEFAULT_CONFIG.encoder.base_channels)
    parser.add_argument("--decoder-base-channels", type=int, default=DEFAULT_CONFIG.decoder.base_channels)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG.training.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG.training.weight_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=DEFAULT_CONFIG.training.grad_clip_norm)
    parser.add_argument("--invisibility-weight", type=float, default=DEFAULT_CONFIG.losses.invisibility_weight)
    parser.add_argument("--detection-weight", type=float, default=DEFAULT_CONFIG.losses.detection_weight)
    parser.add_argument("--ssim-weight", type=float, default=DEFAULT_CONFIG.losses.ssim_weight)
    parser.add_argument("--l2-weight", type=float, default=DEFAULT_CONFIG.losses.l2_weight)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=DEFAULT_CONFIG.training.checkpoint_interval)
    parser.add_argument("--log-interval", type=int, default=DEFAULT_CONFIG.training.log_interval)
    parser.add_argument("--image-log-interval", type=int, default=DEFAULT_CONFIG.training.image_log_interval)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--limit-train-images", type=int, default=None)
    parser.add_argument("--limit-val-images", type=int, default=None)
    parser.add_argument("--limit-test-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.runtime.random_seed)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--disable-attacks", action="store_true")
    parser.add_argument("--run-smoke-test", action="store_true")
    return parser.parse_args()


def create_summary_writer(log_dir: Path) -> Any:
    """Create a TensorBoard writer or a no-op fallback."""

    ensure_directory(log_dir)
    try:
        from torch.utils.tensorboard import SummaryWriter as TensorBoardSummaryWriter
    except ImportError:  # pragma: no cover - optional dependency
        warnings.warn("TensorBoard is not installed; logging will be disabled.", stacklevel=2)
        return NullSummaryWriter()
    return TensorBoardSummaryWriter(log_dir=str(log_dir))


def build_dataloaders(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    """Construct split datasets and dataloaders."""

    datasets = build_split_datasets(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        image_size=args.image_size,
        key_bits=args.key_bits,
        eval_key_seed=args.seed + 1,
        extensions=DEFAULT_CONFIG.data.extensions,
    )
    datasets["train"] = maybe_limit_dataset(datasets["train"], args.limit_train_images, seed=args.seed)
    datasets["val"] = maybe_limit_dataset(datasets["val"], args.limit_val_images, seed=args.seed + 1)
    if "test" in datasets:
        datasets["test"] = maybe_limit_dataset(datasets["test"], args.limit_test_images, seed=args.seed + 2)
    loaders = {
        "train": build_dataloader(
            datasets["train"],
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
            seed=args.seed,
        ),
        "val": build_dataloader(
            datasets["val"],
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            seed=args.seed + 1,
        ),
    }
    if "test" in datasets:
        loaders["test"] = build_dataloader(
            datasets["test"],
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            seed=args.seed + 2,
        )
    return {"datasets": datasets, "loaders": loaders}


def maybe_limit_dataset(
    dataset: Dataset[tuple[Tensor, Tensor]],
    max_images: int | None,
    seed: int,
) -> Dataset[tuple[Tensor, Tensor]]:
    """Return a deterministic subset view when an image limit is requested."""

    if max_images is None or max_images <= 0 or max_images >= len(dataset):
        return dataset

    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_images].tolist()
    return Subset(dataset, indices)


def create_models(
    key_bits: int,
    encoder_alpha: float,
    encoder_base_channels: int,
    decoder_base_channels: int,
    device: torch.device,
) -> tuple[WatermarkEncoder, WatermarkDecoder]:
    """Instantiate the encoder and decoder models."""

    encoder = WatermarkEncoder(
        key_bits=key_bits,
        alpha=encoder_alpha,
        base_channels=encoder_base_channels,
    ).to(device)
    decoder = WatermarkDecoder(
        key_bits=key_bits,
        base_channels=decoder_base_channels,
    ).to(device)
    return encoder, decoder


def is_out_of_memory_error(error: RuntimeError) -> bool:
    """Return whether an exception was caused by OOM."""

    message = str(error).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def move_batch_to_device(batch: tuple[Tensor, Tensor], device: torch.device) -> tuple[Tensor, Tensor]:
    """Move a dataset batch onto the selected device."""

    images, keys = batch
    return images.to(device, non_blocking=True), keys.to(device, non_blocking=True)


def mean_metric(values: list[float]) -> float:
    """Return the arithmetic mean of a metric list."""

    return sum(values) / max(1, len(values))


def run_training_epoch(
    encoder: WatermarkEncoder,
    decoder: WatermarkDecoder,
    augmentor: nn.Module,
    loss_fn: WatermarkLoss,
    optimizer: Adam,
    scaler: torch.amp.GradScaler,
    dataloader: Any,
    device: torch.device,
    args: argparse.Namespace,
    writer: Any,
    global_step: int,
) -> tuple[dict[str, float], int]:
    """Run a single training epoch."""

    encoder.train()
    decoder.train()

    metrics: dict[str, list[float]] = {
        "total_loss": [],
        "invisibility_loss": [],
        "detection_loss": [],
        "bit_accuracy": [],
        "exact_match_accuracy": [],
        "psnr": [],
        "ssim": [],
    }
    progress = tqdm(dataloader, desc="Train", leave=False)
    for batch_index, batch in enumerate(progress, start=1):
        if args.limit_train_batches is not None and batch_index > args.limit_train_batches:
            break

        images, keys = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=device.type == "cuda" and not args.disable_amp):
            watermarked_images, residual = encoder(images, keys, return_residual=True)
            attacked_images = augmentor(watermarked_images)
            decoded_logits = decoder.forward_logits(attacked_images)
            decoded_probs = torch.sigmoid(decoded_logits)
            losses = loss_fn(images, watermarked_images, decoded_logits, keys)

        scaler.scale(losses["total_loss"]).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_norm=args.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_metrics = {
            "total_loss": float(losses["total_loss"].detach().item()),
            "invisibility_loss": float(losses["invisibility_loss"].detach().item()),
            "detection_loss": float(losses["detection_loss"].detach().item()),
            "bit_accuracy": float(bit_accuracy_from_probs(decoded_probs.detach(), keys).item()),
            "exact_match_accuracy": float(exact_match_accuracy_from_probs(decoded_probs.detach(), keys).item()),
            "psnr": float(peak_signal_to_noise_ratio(images.detach(), watermarked_images.detach()).item()),
            "ssim": float(structural_similarity(images.detach(), watermarked_images.detach()).item()),
        }
        for name, value in batch_metrics.items():
            metrics[name].append(value)

        if batch_index % args.log_interval == 0:
            writer.add_scalar("train/total_loss_step", batch_metrics["total_loss"], global_step)
            writer.add_scalar("train/bit_accuracy_step", batch_metrics["bit_accuracy"], global_step)

        if global_step % args.image_log_interval == 0:
            writer.add_image(
                "train/original",
                make_image_grid(images.detach(), max_images=DEFAULT_CONFIG.training.tensorboard_image_count),
                global_step,
            )
            writer.add_image(
                "train/watermarked",
                make_image_grid(watermarked_images.detach(), max_images=DEFAULT_CONFIG.training.tensorboard_image_count),
                global_step,
            )
            residual_viz = (residual.detach() * 10.0 + 0.5).clamp(0.0, 1.0)
            writer.add_image(
                "train/residual_x10",
                make_image_grid(residual_viz, max_images=DEFAULT_CONFIG.training.tensorboard_image_count),
                global_step,
            )

        progress.set_postfix(
            loss=f"{batch_metrics['total_loss']:.4f}",
            bit_acc=f"{batch_metrics['bit_accuracy']:.4f}",
        )
        global_step += 1

    return {name: mean_metric(values) for name, values in metrics.items()}, global_step


@torch.no_grad()
def run_validation_epoch(
    encoder: WatermarkEncoder,
    decoder: WatermarkDecoder,
    loss_fn: WatermarkLoss,
    dataloader: Any,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Evaluate the model on the validation split without attacks."""

    encoder.eval()
    decoder.eval()

    metrics: dict[str, list[float]] = {
        "total_loss": [],
        "invisibility_loss": [],
        "detection_loss": [],
        "bit_accuracy": [],
        "exact_match_accuracy": [],
        "psnr": [],
        "ssim": [],
    }
    progress = tqdm(dataloader, desc="Val", leave=False)
    for batch_index, batch in enumerate(progress, start=1):
        if args.limit_val_batches is not None and batch_index > args.limit_val_batches:
            break

        images, keys = move_batch_to_device(batch, device)
        watermarked_images = encoder(images, keys)
        decoded_logits = decoder.forward_logits(watermarked_images)
        decoded_probs = torch.sigmoid(decoded_logits)
        losses = loss_fn(images, watermarked_images, decoded_logits, keys)

        batch_metrics = {
            "total_loss": float(losses["total_loss"].item()),
            "invisibility_loss": float(losses["invisibility_loss"].item()),
            "detection_loss": float(losses["detection_loss"].item()),
            "bit_accuracy": float(bit_accuracy_from_probs(decoded_probs, keys).item()),
            "exact_match_accuracy": float(exact_match_accuracy_from_probs(decoded_probs, keys).item()),
            "psnr": float(peak_signal_to_noise_ratio(images, watermarked_images).item()),
            "ssim": float(structural_similarity(images, watermarked_images).item()),
        }
        for name, value in batch_metrics.items():
            metrics[name].append(value)
        progress.set_postfix(bit_acc=f"{batch_metrics['bit_accuracy']:.4f}")

    return {name: mean_metric(values) for name, values in metrics.items()}


def checkpoint_payload(
    epoch: int,
    encoder: WatermarkEncoder,
    decoder: WatermarkDecoder,
    optimizer: Adam,
    scheduler: ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    best_val_bit_accuracy: float,
    global_step: int,
) -> dict[str, Any]:
    """Build the checkpoint payload stored on disk."""

    return {
        "epoch": epoch,
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_bit_accuracy": best_val_bit_accuracy,
        "global_step": global_step,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "key_bits": args.key_bits,
        "image_size": args.image_size,
        "seed": args.seed,
        "encoder_alpha": args.encoder_alpha,
        "encoder_base_channels": args.encoder_base_channels,
        "decoder_base_channels": args.decoder_base_channels,
    }


def maybe_resume_training(
    args: argparse.Namespace,
    encoder: WatermarkEncoder,
    decoder: WatermarkDecoder,
    optimizer: Adam,
    scheduler: ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, float, int]:
    """Load a checkpoint when resume is requested."""

    checkpoint_path = args.resume
    if checkpoint_path is None and args.auto_resume:
        checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
    if checkpoint_path is None:
        return 1, 0.0, 0

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    args.train_batch_size = int(checkpoint.get("train_batch_size", args.train_batch_size))
    args.eval_batch_size = int(checkpoint.get("eval_batch_size", args.eval_batch_size))
    start_epoch = int(checkpoint["epoch"]) + 1
    best_val_bit_accuracy = float(checkpoint.get("best_val_bit_accuracy", 0.0))
    global_step = int(checkpoint.get("global_step", 0))

    print(f"Resumed from checkpoint: {Path(checkpoint_path).expanduser().resolve()}")
    return start_epoch, best_val_bit_accuracy, global_step


def maybe_adopt_model_hparams_from_resume(args: argparse.Namespace) -> None:
    """Adopt checkpoint-defined model hyperparameters before model construction."""

    checkpoint_path = args.resume
    if checkpoint_path is None and args.auto_resume:
        checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
    if checkpoint_path is None:
        return

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    args.key_bits = int(checkpoint.get("key_bits", args.key_bits))
    args.encoder_alpha = float(checkpoint.get("encoder_alpha", args.encoder_alpha))
    args.encoder_base_channels = int(checkpoint.get("encoder_base_channels", args.encoder_base_channels))
    args.decoder_base_channels = int(checkpoint.get("decoder_base_channels", args.decoder_base_channels))


def print_dataset_summary(split_name: str, dataset: Any, batch_size: int) -> None:
    """Print a concise summary for a dataset split."""

    print(f"{split_name}: {len(dataset)} images | batch_size={batch_size}")


def run_smoke_test(args: argparse.Namespace) -> None:
    """Run a one-batch training and validation sanity pass."""

    args.limit_train_batches = 1
    args.limit_val_batches = 1
    args.epochs = 1
    train(args)


def train(args: argparse.Namespace) -> None:
    """Run the full training workflow."""

    set_global_seed(args.seed, deterministic_algorithms=DEFAULT_CONFIG.runtime.deterministic_algorithms)
    device, device_name = get_best_device()
    print(f"Using device: {device_name}")

    ensure_directory(args.checkpoint_dir)
    writer = create_summary_writer(args.log_dir)
    maybe_adopt_model_hparams_from_resume(args)
    dataloader_bundle = build_dataloaders(args, device)
    datasets = dataloader_bundle["datasets"]
    loaders = dataloader_bundle["loaders"]

    print_dataset_summary("train", datasets["train"], args.train_batch_size)
    print_dataset_summary("val", datasets["val"], args.eval_batch_size)
    if "test" in datasets:
        print_dataset_summary("test", datasets["test"], args.eval_batch_size)

    encoder, decoder = create_models(
        key_bits=args.key_bits,
        encoder_alpha=args.encoder_alpha,
        encoder_base_channels=args.encoder_base_channels,
        decoder_base_channels=args.decoder_base_channels,
        device=device,
    )
    augmentor = RandomAugmentationPipeline(enabled=not args.disable_attacks).to(device)
    loss_fn = WatermarkLoss(
        invisibility_weight=args.invisibility_weight,
        detection_weight=args.detection_weight,
        ssim_weight=args.ssim_weight,
        l2_weight=args.l2_weight,
    ).to(device)
    print(f"Training attacks enabled: {not args.disable_attacks}")
    print(
        "Loss weights: "
        f"invisibility={args.invisibility_weight}, "
        f"detection={args.detection_weight}, "
        f"ssim={args.ssim_weight}, "
        f"l2={args.l2_weight}"
    )
    print(f"Encoder alpha: {args.encoder_alpha}")
    print(
        "Model capacity: "
        f"encoder_base_channels={args.encoder_base_channels}, "
        f"decoder_base_channels={args.decoder_base_channels}"
    )
    optimizer = Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=DEFAULT_CONFIG.training.lr_factor,
        patience=DEFAULT_CONFIG.training.lr_patience,
    )
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda" and not args.disable_amp)

    start_epoch, best_val_bit_accuracy, global_step = maybe_resume_training(
        args,
        encoder,
        decoder,
        optimizer,
        scheduler,
        scaler,
        device,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_retry_count = 0
        while True:
            try:
                epoch_start = time.time()
                train_metrics, global_step = run_training_epoch(
                    encoder=encoder,
                    decoder=decoder,
                    augmentor=augmentor,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    scaler=scaler,
                    dataloader=loaders["train"],
                    device=device,
                    args=args,
                    writer=writer,
                    global_step=global_step,
                )
                break
            except RuntimeError as error:
                if (
                    not is_out_of_memory_error(error)
                    or args.train_batch_size <= 1
                    or epoch_retry_count >= DEFAULT_CONFIG.training.max_batch_size_retries
                ):
                    raise

                epoch_retry_count += 1
                new_batch_size = max(1, args.train_batch_size // 2)
                warnings.warn(
                    f"Encountered OOM at epoch {epoch}. Reducing train batch size from "
                    f"{args.train_batch_size} to {new_batch_size} and retrying.",
                    stacklevel=2,
                )
                args.train_batch_size = new_batch_size
                args.eval_batch_size = min(args.eval_batch_size, args.train_batch_size)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                dataloader_bundle = build_dataloaders(args, device)
                loaders = dataloader_bundle["loaders"]

        val_metrics = run_validation_epoch(
            encoder=encoder,
            decoder=decoder,
            loss_fn=loss_fn,
            dataloader=loaders["val"],
            device=device,
            args=args,
        )
        scheduler.step(val_metrics["bit_accuracy"])

        epoch_duration = time.time() - epoch_start
        for name, value in train_metrics.items():
            writer.add_scalar(f"train/{name}", value, epoch)
        for name, value in val_metrics.items():
            writer.add_scalar(f"val/{name}", value, epoch)
        writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], epoch)
        history_path = append_epoch_history(
            log_dir=args.log_dir,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_metrics['total_loss']:.4f} | "
            f"train_bit_acc={train_metrics['bit_accuracy']:.4f} | "
            f"val_loss={val_metrics['total_loss']:.4f} | "
            f"val_bit_acc={val_metrics['bit_accuracy']:.4f} | "
            f"val_psnr={val_metrics['psnr']:.2f} | "
            f"val_ssim={val_metrics['ssim']:.4f} | "
            f"time={epoch_duration:.1f}s"
        )

        improved = val_metrics["bit_accuracy"] >= best_val_bit_accuracy
        if val_metrics["bit_accuracy"] >= best_val_bit_accuracy:
            best_val_bit_accuracy = val_metrics["bit_accuracy"]
        state = checkpoint_payload(
            epoch=epoch,
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            args=args,
            best_val_bit_accuracy=best_val_bit_accuracy,
            global_step=global_step,
        )
        latest_checkpoint = save_checkpoint(state, args.checkpoint_dir / "latest.pt")
        if epoch % args.save_every == 0:
            save_checkpoint(state, args.checkpoint_dir / f"epoch_{epoch:03d}.pt")
        if improved:
            save_checkpoint(state, args.checkpoint_dir / "best.pt")
            print(f"Saved new best checkpoint: {args.checkpoint_dir / 'best.pt'}")

        print(f"Updated latest checkpoint: {latest_checkpoint}")
        print(f"Updated metrics history: {history_path}")

    writer.close()


def main() -> None:
    """Program entry point."""

    args = parse_args()
    if args.run_smoke_test:
        run_smoke_test(args)
        return
    train(args)


if __name__ == "__main__":
    main()
