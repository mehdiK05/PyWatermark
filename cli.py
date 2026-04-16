"""Command-line interface for embedding and detecting PyWatermark watermarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import DEFAULT_CONFIG
from utils import (
    binary_string_to_tensor,
    bit_accuracy_from_probs,
    get_best_device,
    load_image_tensor,
    load_models_from_checkpoint,
    peak_signal_to_noise_ratio,
    save_image_tensor,
    set_global_seed,
    structural_similarity,
    tensor_to_binary_string,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the PyWatermark CLI."""

    parser = argparse.ArgumentParser(description="Embed or detect invisible watermarks in images.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    embed_parser = subparsers.add_parser("embed", help="Embed a watermark into an image.")
    embed_parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CONFIG.paths.checkpoint_dir / "best.pt")
    embed_parser.add_argument("--input", type=Path, required=True)
    embed_parser.add_argument("--output", type=Path, required=True)
    embed_parser.add_argument("--key", type=str, default=None)
    embed_parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.runtime.random_seed)

    detect_parser = subparsers.add_parser("detect", help="Decode watermark bits from an image.")
    detect_parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CONFIG.paths.checkpoint_dir / "best.pt")
    detect_parser.add_argument("--input", type=Path, required=True)
    detect_parser.add_argument("--key", type=str, default=None)
    detect_parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIG.evaluation.threshold)
    detect_parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.runtime.random_seed)

    return parser.parse_args()


def resolve_key(key: str | None, key_bits: int, seed: int) -> torch.Tensor:
    """Return a binary key tensor from a provided string or a generated random key."""

    if key is not None:
        key_tensor = binary_string_to_tensor(key)
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)
        key_tensor = torch.randint(0, 2, (key_bits,), generator=generator, dtype=torch.int64).to(torch.float32)

    if key_tensor.numel() != key_bits:
        raise ValueError(f"Expected a key with {key_bits} bits, received {key_tensor.numel()}.")
    return key_tensor


def embed_image(args: argparse.Namespace) -> None:
    """Embed a watermark key into a single image and save the result."""

    set_global_seed(args.seed, deterministic_algorithms=False)
    device, device_name = get_best_device()
    encoder, decoder, checkpoint = load_models_from_checkpoint(args.checkpoint, device)
    image_size = int(checkpoint.get("image_size", DEFAULT_CONFIG.data.image_size))
    key_bits = int(checkpoint.get("key_bits", DEFAULT_CONFIG.data.key_bits))

    input_image = load_image_tensor(args.input, image_size=image_size).to(device)
    key_tensor = resolve_key(args.key, key_bits=key_bits, seed=args.seed).to(device).unsqueeze(0)

    with torch.no_grad():
        watermarked_image = encoder(input_image, key_tensor)
        decoded_probs = decoder(watermarked_image)

    save_path = save_image_tensor(watermarked_image, args.output)
    psnr = peak_signal_to_noise_ratio(input_image, watermarked_image).item()
    ssim = structural_similarity(input_image, watermarked_image).item()
    bit_accuracy = bit_accuracy_from_probs(decoded_probs, key_tensor).item()

    print(f"Device: {device_name}")
    print(f"Embedded key: {tensor_to_binary_string(key_tensor)}")
    print(f"Decoded key:   {tensor_to_binary_string(decoded_probs)}")
    print(f"Bit accuracy:  {bit_accuracy:.4f}")
    print(f"PSNR:          {psnr:.2f} dB")
    print(f"SSIM:          {ssim:.4f}")
    print(f"Saved image:   {save_path.resolve()}")


def detect_image(args: argparse.Namespace) -> None:
    """Decode watermark bits from a single image."""

    set_global_seed(args.seed, deterministic_algorithms=False)
    device, device_name = get_best_device()
    _encoder, decoder, checkpoint = load_models_from_checkpoint(args.checkpoint, device)
    image_size = int(checkpoint.get("image_size", DEFAULT_CONFIG.data.image_size))
    key_bits = int(checkpoint.get("key_bits", DEFAULT_CONFIG.data.key_bits))

    input_image = load_image_tensor(args.input, image_size=image_size).to(device)
    expected_key = resolve_key(args.key, key_bits=key_bits, seed=args.seed).to(device).unsqueeze(0) if args.key else None

    with torch.no_grad():
        probabilities = decoder(input_image)

    print(f"Device: {device_name}")
    print(f"Predicted key: {tensor_to_binary_string(probabilities, threshold=args.threshold)}")
    print(f"Probabilities: {[round(float(value), 4) for value in probabilities.squeeze(0).cpu()]}")

    if expected_key is not None:
        bit_accuracy = bit_accuracy_from_probs(probabilities, expected_key, threshold=args.threshold).item()
        per_bit_matches = (
            (probabilities >= args.threshold).to(expected_key.dtype).squeeze(0) == expected_key.squeeze(0)
        ).to(torch.int64)
        print(f"Expected key:  {tensor_to_binary_string(expected_key, threshold=args.threshold)}")
        print(f"Bit accuracy:  {bit_accuracy:.4f}")
        print(f"Per-bit match: {per_bit_matches.cpu().tolist()}")


def main() -> None:
    """Program entry point."""

    args = parse_args()
    if args.command == "embed":
        embed_image(args)
    else:
        detect_image(args)


if __name__ == "__main__":
    main()
