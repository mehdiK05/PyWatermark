"""Model construction and checkpoint loading helpers for PyWatermark."""

from __future__ import annotations

from pathlib import Path

import torch

from config import DEFAULT_CONFIG
from models.decoder import WatermarkDecoder
from models.encoder import WatermarkEncoder
from utils.checkpoint import load_checkpoint


def load_models_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    key_bits: int | None = None,
) -> tuple[WatermarkEncoder, WatermarkDecoder, dict]:
    """Instantiate models, load checkpoint weights, and return the checkpoint payload."""

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    resolved_key_bits = int(checkpoint.get("key_bits", key_bits if key_bits is not None else 0))
    if resolved_key_bits <= 0:
        raise ValueError("key_bits could not be determined from the checkpoint or arguments.")

    encoder = WatermarkEncoder(
        key_bits=resolved_key_bits,
        alpha=float(checkpoint.get("encoder_alpha", DEFAULT_CONFIG.encoder.alpha)),
        base_channels=int(checkpoint.get("encoder_base_channels", DEFAULT_CONFIG.encoder.base_channels)),
        image_channels=DEFAULT_CONFIG.encoder.image_channels,
    ).to(device)
    decoder = WatermarkDecoder(
        key_bits=resolved_key_bits,
        image_channels=DEFAULT_CONFIG.decoder.image_channels,
        base_channels=int(checkpoint.get("decoder_base_channels", DEFAULT_CONFIG.decoder.base_channels)),
        residual_blocks=DEFAULT_CONFIG.decoder.residual_blocks,
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder, checkpoint
