"""Lightweight ResNet-style watermark decoder for PyWatermark."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from config import DEFAULT_CONFIG


class ResidualBlock(nn.Module):
    """A compact residual block used by the watermark decoder."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply residual feature refinement."""

        return self.activation(inputs + self.block(inputs))


class DownsampleBlock(nn.Module):
    """Reduce spatial resolution while increasing channel capacity."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Downsample a feature map by a factor of two."""

        return self.block(inputs)


class WatermarkDecoder(nn.Module):
    """Decode watermark bits from a possibly attacked image."""

    def __init__(
        self,
        key_bits: int = DEFAULT_CONFIG.data.key_bits,
        image_channels: int = DEFAULT_CONFIG.decoder.image_channels,
        base_channels: int = DEFAULT_CONFIG.decoder.base_channels,
        residual_blocks: int = DEFAULT_CONFIG.decoder.residual_blocks,
    ) -> None:
        super().__init__()
        if key_bits <= 0:
            raise ValueError("key_bits must be positive.")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive.")
        if residual_blocks != 4:
            raise ValueError("This decoder is configured for exactly 4 residual blocks.")

        self.key_bits = key_bits
        self.image_channels = image_channels

        self.stem = nn.Sequential(
            nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(base_channels),
                ResidualBlock(base_channels),
                ResidualBlock(base_channels * 2),
                ResidualBlock(base_channels * 2),
            ]
        )
        self.downsample_after_block_2 = DownsampleBlock(base_channels, base_channels * 2)
        self.downsample_after_block_4 = DownsampleBlock(base_channels * 2, base_channels * 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels * 4, key_bits)

    def forward_logits(self, image: Tensor) -> Tensor:
        """Return raw decoder logits for BCE-based optimization."""

        self._validate_inputs(image)
        features = self.stem(image)
        features = self.residual_blocks[0](features)
        features = self.residual_blocks[1](features)
        features = self.downsample_after_block_2(features)
        features = self.residual_blocks[2](features)
        features = self.residual_blocks[3](features)
        features = self.downsample_after_block_4(features)
        pooled = self.pool(features).flatten(1)
        return self.fc(pooled)

    def forward(self, image: Tensor) -> Tensor:
        """Return sigmoid probabilities for each decoded watermark bit."""

        return torch.sigmoid(self.forward_logits(image))

    def _validate_inputs(self, image: Tensor) -> None:
        """Validate input image shape before inference."""

        if image.ndim != 4:
            raise ValueError(f"image must have shape (B, C, H, W); received {tuple(image.shape)}.")
        if image.shape[1] != self.image_channels:
            raise ValueError(
                f"image must have {self.image_channels} channels; received {image.shape[1]}."
            )


def _run_smoke_test() -> None:
    """Run a lightweight decoder validation locally."""

    torch.manual_seed(0)

    decoder = WatermarkDecoder()
    decoder.eval()

    batch_size = 2
    image_size = DEFAULT_CONFIG.data.image_size
    image = torch.rand(batch_size, DEFAULT_CONFIG.data.image_channels, image_size, image_size)

    with torch.no_grad():
        logits = decoder.forward_logits(image)
        probabilities = decoder(image)

    assert logits.shape == (batch_size, DEFAULT_CONFIG.data.key_bits)
    assert probabilities.shape == (batch_size, DEFAULT_CONFIG.data.key_bits)
    assert torch.all(probabilities >= 0.0) and torch.all(probabilities <= 1.0)
    print(f"Decoder logits shape: {tuple(logits.shape)}")
    print("Decoder smoke test passed.")


if __name__ == "__main__":
    _run_smoke_test()
