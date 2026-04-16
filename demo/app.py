"""Gradio demo for the PyWatermark embed and detect workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from config import DEFAULT_CONFIG
from utils import (
    binary_string_to_tensor,
    bit_accuracy_from_probs,
    get_best_device,
    load_models_from_checkpoint,
    peak_signal_to_noise_ratio,
    pil_image_to_tensor,
    set_global_seed,
    structural_similarity,
    tensor_to_binary_string,
    tensor_to_pil_image,
)

try:
    import gradio as gr
except ImportError:  # pragma: no cover - optional dependency
    gr = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the demo."""

    parser = argparse.ArgumentParser(description="Launch the PyWatermark Gradio demo.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CONFIG.paths.checkpoint_dir / "best.pt")
    parser.set_defaults(share=DEFAULT_CONFIG.demo.share)
    parser.add_argument("--share", dest="share", action="store_true")
    parser.add_argument("--no-share", dest="share", action="store_false")
    parser.add_argument("--server-name", type=str, default=DEFAULT_CONFIG.demo.server_name)
    parser.add_argument("--server-port", type=int, default=DEFAULT_CONFIG.demo.server_port)
    return parser.parse_args()


def prepare_image(image: Image.Image, image_size: int, device: torch.device) -> torch.Tensor:
    """Convert a PIL image into a model-ready tensor batch."""

    resized = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
    return pil_image_to_tensor(resized).unsqueeze(0).to(device)


def create_app(checkpoint_path: str | Path) -> "gr.Blocks":
    """Build the Gradio demo application."""

    if gr is None:
        raise ImportError("Gradio is not installed. Run `pip install -r requirements.txt` first.")

    device, device_name = get_best_device()
    encoder, decoder, checkpoint = load_models_from_checkpoint(checkpoint_path, device)
    image_size = int(checkpoint.get("image_size", DEFAULT_CONFIG.data.image_size))
    key_bits = int(checkpoint.get("key_bits", DEFAULT_CONFIG.data.key_bits))

    def embed(image: Image.Image | None, key: str, seed: int) -> tuple[Image.Image | None, str, str]:
        """Embed a watermark into an uploaded image."""

        if image is None:
            return None, "", "Upload an image first."

        set_global_seed(seed, deterministic_algorithms=False)
        image_tensor = prepare_image(image, image_size=image_size, device=device)
        if key.strip():
            key_tensor = binary_string_to_tensor(key.strip())
        else:
            generator = torch.Generator()
            generator.manual_seed(seed)
            key_tensor = torch.randint(0, 2, (key_bits,), generator=generator, dtype=torch.int64).to(torch.float32)
        if key_tensor.numel() != key_bits:
            raise ValueError(f"Expected {key_bits} bits, received {key_tensor.numel()}.")
        key_batch = key_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            watermarked = encoder(image_tensor, key_batch)
            decoded_probs = decoder(watermarked)

        output_image = tensor_to_pil_image(watermarked)
        metrics = "\n".join(
            [
                f"Device: {device_name}",
                f"Embedded key: {tensor_to_binary_string(key_batch)}",
                f"Decoded key: {tensor_to_binary_string(decoded_probs)}",
                f"Bit accuracy: {bit_accuracy_from_probs(decoded_probs, key_batch).item():.4f}",
                f"PSNR: {peak_signal_to_noise_ratio(image_tensor, watermarked).item():.2f} dB",
                f"SSIM: {structural_similarity(image_tensor, watermarked).item():.4f}",
            ]
        )
        return output_image, tensor_to_binary_string(key_batch), metrics

    def detect(image: Image.Image | None, expected_key: str, threshold: float) -> tuple[str, str]:
        """Decode watermark bits from an uploaded image."""

        if image is None:
            return "", "Upload an image first."

        image_tensor = prepare_image(image, image_size=image_size, device=device)
        with torch.no_grad():
            decoded_probs = decoder(image_tensor)

        report_lines = [
            f"Device: {device_name}",
            f"Predicted key: {tensor_to_binary_string(decoded_probs, threshold=threshold)}",
            f"Probabilities: {[round(float(value), 4) for value in decoded_probs.squeeze(0).cpu()]}",
        ]
        if expected_key.strip():
            key_tensor = binary_string_to_tensor(expected_key.strip())
            if key_tensor.numel() != key_bits:
                raise ValueError(f"Expected {key_bits} bits, received {key_tensor.numel()}.")
            key_batch = key_tensor.unsqueeze(0).to(device)
            report_lines.append(
                f"Bit accuracy vs expected key: {bit_accuracy_from_probs(decoded_probs, key_batch, threshold=threshold).item():.4f}"
            )
        return tensor_to_binary_string(decoded_probs, threshold=threshold), "\n".join(report_lines)

    with gr.Blocks(title=DEFAULT_CONFIG.project_name) as demo:
        gr.Markdown(
            f"# {DEFAULT_CONFIG.project_name}\n"
            "Embed and detect invisible watermarks using a trained PyTorch checkpoint."
        )
        with gr.Tab("Embed"):
            with gr.Row():
                embed_input = gr.Image(type="pil", label="Input Image")
                embed_output = gr.Image(type="pil", label="Watermarked Image")
            embed_key = gr.Textbox(label=f"Watermark Key ({key_bits} bits, optional)")
            embed_seed = gr.Number(label="Random Seed", value=DEFAULT_CONFIG.runtime.random_seed, precision=0)
            embed_key_output = gr.Textbox(label="Embedded Key")
            embed_report = gr.Textbox(label="Embed Report", lines=6)
            gr.Button("Embed").click(
                embed,
                inputs=[embed_input, embed_key, embed_seed],
                outputs=[embed_output, embed_key_output, embed_report],
            )

        with gr.Tab("Detect"):
            detect_input = gr.Image(type="pil", label="Input Image")
            detect_key = gr.Textbox(label=f"Expected Key ({key_bits} bits, optional)")
            detect_threshold = gr.Slider(
                label="Detection Threshold",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=DEFAULT_CONFIG.evaluation.threshold,
            )
            detect_key_output = gr.Textbox(label="Predicted Key")
            detect_report = gr.Textbox(label="Detection Report", lines=8)
            gr.Button("Detect").click(
                detect,
                inputs=[detect_input, detect_key, detect_threshold],
                outputs=[detect_key_output, detect_report],
            )

    return demo


def main() -> None:
    """Launch the Gradio application."""

    args = parse_args()
    app = create_app(args.checkpoint)
    app.launch(share=args.share, server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
