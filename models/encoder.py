"""UNet-lite watermark encoder for PyWatermark."""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from config import DEFAULT_CONFIG

_NUM_DOWNSAMPLE_BLOCKS: Final[int] = 3


class DownsampleBlock(nn.Module):
    """Apply feature extraction followed by spatial downsampling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Return the pooled tensor and the pre-pool skip features."""

        features = self.activation(self.norm(self.conv(inputs)))
        return self.pool(features), features


class UpsampleBlock(nn.Module):
    """Upsample bottleneck features and fuse them with a skip connection."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(
                out_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: Tensor, skip: Tensor) -> Tensor:
        """Upsample a feature map and merge it with its paired skip tensor."""

        upsampled = self.up(inputs, output_size=skip.size())
        if upsampled.shape[-2:] != skip.shape[-2:]:
            upsampled = F.interpolate(
                upsampled,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        merged = torch.cat((upsampled, skip), dim=1)
        return self.fuse(merged)


class WatermarkEncoder(nn.Module):
    """Encode a watermark key into an RGB image via a learned residual."""

    def __init__(
        self,
        key_bits: int = DEFAULT_CONFIG.data.key_bits,
        alpha: float = DEFAULT_CONFIG.encoder.alpha,
        base_channels: int = DEFAULT_CONFIG.encoder.base_channels,
        image_channels: int = DEFAULT_CONFIG.encoder.image_channels,
    ) -> None:
        super().__init__()
        if key_bits <= 0:
            raise ValueError("key_bits must be a positive integer.")
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if base_channels <= 0:
            raise ValueError("base_channels must be a positive integer.")
        if image_channels <= 0:
            raise ValueError("image_channels must be a positive integer.")

        self.key_bits = key_bits
        self.alpha = alpha
        self.base_channels = base_channels
        self.image_channels = image_channels

        encoder_channels = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
        )

        self.down_blocks = nn.ModuleList(
            [
                DownsampleBlock(image_channels, encoder_channels[0]),
                DownsampleBlock(encoder_channels[0], encoder_channels[1]),
                DownsampleBlock(encoder_channels[1], encoder_channels[2]),
            ]
        )

        self.up_blocks = nn.ModuleList(
            [
                UpsampleBlock(
                    encoder_channels[2] + key_bits,
                    encoder_channels[2],
                    encoder_channels[2],
                ),
                UpsampleBlock(
                    encoder_channels[2],
                    encoder_channels[1],
                    encoder_channels[1],
                ),
                UpsampleBlock(
                    encoder_channels[1],
                    encoder_channels[0],
                    encoder_channels[0],
                ),
            ]
        )

        self.residual_head = nn.Conv2d(
            encoder_channels[0],
            image_channels,
            kernel_size=1,
        )

    def forward(
        self,
        image: Tensor,
        key: Tensor,
        return_residual: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Return a clamped watermarked image and optionally its residual."""

        residual = self.compute_residual(image, key)
        watermarked_image = torch.clamp(image + residual, min=0.0, max=1.0)
        if return_residual:
            return watermarked_image, residual
        return watermarked_image

    def compute_residual(self, image: Tensor, key: Tensor) -> Tensor:
        """Compute the bounded watermark residual before image clamping."""

        self._validate_inputs(image, key)

        features = image
        skip_connections: list[Tensor] = []
        for down_block in self.down_blocks:
            features, skip = down_block(features)
            skip_connections.append(skip)

        bottleneck = self._inject_key(features, key)
        decoded = bottleneck
        for up_block, skip in zip(self.up_blocks, reversed(skip_connections)):
            decoded = up_block(decoded, skip)

        return torch.tanh(self.residual_head(decoded)) * self.alpha

    def _inject_key(self, bottleneck: Tensor, key: Tensor) -> Tensor:
        """Broadcast the key spatially and concatenate it at the bottleneck."""

        key_map = key.to(device=bottleneck.device, dtype=bottleneck.dtype)
        key_map = key_map.unsqueeze(-1).unsqueeze(-1)
        key_map = key_map.expand(-1, -1, bottleneck.shape[-2], bottleneck.shape[-1])
        return torch.cat((bottleneck, key_map), dim=1)

    def _validate_inputs(self, image: Tensor, key: Tensor) -> None:
        """Validate public forward inputs before running the network."""

        if image.ndim != 4:
            raise ValueError(
                f"image must have shape (B, C, H, W); received {tuple(image.shape)}."
            )
        if key.ndim != 2:
            raise ValueError(f"key must have shape (B, N); received {tuple(key.shape)}.")
        if image.shape[0] != key.shape[0]:
            raise ValueError(
                "image and key batch sizes must match; "
                f"received {image.shape[0]} and {key.shape[0]}."
            )
        if image.shape[1] != self.image_channels:
            raise ValueError(
                f"image must have {self.image_channels} channels; "
                f"received {image.shape[1]}."
            )
        if key.shape[1] != self.key_bits:
            raise ValueError(
                f"key must contain {self.key_bits} bits; received {key.shape[1]}."
            )
        minimum_spatial_size = 2**_NUM_DOWNSAMPLE_BLOCKS
        if image.shape[-2] < minimum_spatial_size or image.shape[-1] < minimum_spatial_size:
            raise ValueError(
                "image height and width must both be at least "
                f"{minimum_spatial_size} pixels."
            )


def _run_smoke_test() -> None:
    """Run a lightweight local validation of the encoder implementation."""

    torch.manual_seed(0)

    encoder = WatermarkEncoder()
    encoder.eval()

    batch_size = 2
    image_size = DEFAULT_CONFIG.data.image_size
    image_channels = DEFAULT_CONFIG.encoder.image_channels
    key_bits = DEFAULT_CONFIG.data.key_bits

    image = torch.rand(batch_size, image_channels, image_size, image_size)
    key = torch.randint(0, 2, (batch_size, key_bits), dtype=torch.int64).to(torch.float32)

    with torch.no_grad():
        watermarked_image, residual = encoder(image, key, return_residual=True)

    residual_min = residual.min().item()
    residual_max = residual.max().item()
    residual_abs_max = residual.abs().max().item()
    residual_mean = residual.mean().item()
    residual_std = residual.std().item()

    print(f"Input shape: {tuple(image.shape)}")
    print(f"Output shape: {tuple(watermarked_image.shape)}")
    print(
        "Residual stats: "
        f"min={residual_min:.4f}, "
        f"max={residual_max:.4f}, "
        f"abs_max={residual_abs_max:.4f}, "
        f"mean={residual_mean:.4f}, "
        f"std={residual_std:.4f}"
    )

    assert watermarked_image.shape == image.shape, "Output shape must match the input shape."
    assert watermarked_image.min().item() >= 0.0, "Watermarked output must be clamped."
    assert watermarked_image.max().item() <= 1.0, "Watermarked output must be clamped."
    assert residual_abs_max <= encoder.alpha + 1e-6, "Residual magnitude must be alpha-bounded."

    print("Encoder smoke test passed.")


if __name__ == "__main__":
    _run_smoke_test()
