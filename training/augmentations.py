"""Differentiable robustness augmentations for PyWatermark."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from config import DEFAULT_CONFIG

try:
    import kornia.augmentation as K  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    K = None

AttackFn = Callable[[Tensor], Tensor]


def _ste_round(values: Tensor) -> Tensor:
    """Apply straight-through rounding."""

    return values + (torch.round(values) - values).detach()


def apply_differentiable_jpeg(image: Tensor, quality: int) -> Tensor:
    """Approximate JPEG compression with differentiable resizing and quantization."""

    compression_strength = float(max(1, min(100, 100 - quality))) / 100.0
    scale = max(0.40, 1.0 - 0.55 * compression_strength)
    height, width = image.shape[-2:]
    resized_height = max(8, int(round(height * scale)))
    resized_width = max(8, int(round(width * scale)))

    compressed = F.interpolate(
        image,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )
    compressed = F.interpolate(
        compressed,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )

    quantization_levels = max(8, int(round(255 * (1.0 - 0.90 * compression_strength))))
    compressed = _ste_round(compressed * quantization_levels) / quantization_levels
    return compressed.clamp(0.0, 1.0)


def apply_gaussian_blur(image: Tensor, kernel_size: int, sigma: float) -> Tensor:
    """Apply differentiable depthwise Gaussian blur."""

    radius = kernel_size // 2
    positions = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel_1d = torch.exp(-(positions**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.expand(image.shape[1], 1, kernel_size, kernel_size).contiguous()
    return F.conv2d(image, kernel, padding=radius, groups=image.shape[1])


def apply_random_crop_resize(
    image: Tensor,
    scale_min: float,
    scale_max: float,
) -> Tensor:
    """Apply differentiable crop and resize using affine sampling."""

    batch_size = image.shape[0]
    scales = torch.empty(batch_size, device=image.device).uniform_(scale_min, scale_max)
    max_offsets = (1.0 - scales).clamp_min(0.0)
    offsets_x = (torch.rand(batch_size, device=image.device) * 2.0 - 1.0) * max_offsets
    offsets_y = (torch.rand(batch_size, device=image.device) * 2.0 - 1.0) * max_offsets

    theta = torch.zeros(batch_size, 2, 3, device=image.device, dtype=image.dtype)
    theta[:, 0, 0] = scales
    theta[:, 1, 1] = scales
    theta[:, 0, 2] = offsets_x
    theta[:, 1, 2] = offsets_y

    grid = F.affine_grid(theta, size=image.size(), align_corners=False)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=False)


def apply_color_jitter(
    image: Tensor,
    brightness_jitter: float,
    contrast_jitter: float,
    saturation_jitter: float,
    hue_jitter: float,
) -> Tensor:
    """Apply differentiable color jitter, preferring Kornia when available."""

    if K is not None:
        augmenter = K.ColorJiggle(
            brightness=brightness_jitter,
            contrast=contrast_jitter,
            saturation=saturation_jitter,
            hue=hue_jitter,
            p=1.0,
        )
        return augmenter(image).clamp(0.0, 1.0)

    brightness_scale = torch.empty(image.shape[0], 1, 1, 1, device=image.device, dtype=image.dtype).uniform_(
        max(0.0, 1.0 - brightness_jitter),
        1.0 + brightness_jitter,
    )
    contrast_scale = torch.empty(image.shape[0], 1, 1, 1, device=image.device, dtype=image.dtype).uniform_(
        max(0.0, 1.0 - contrast_jitter),
        1.0 + contrast_jitter,
    )
    saturation_scale = torch.empty(image.shape[0], 1, 1, 1, device=image.device, dtype=image.dtype).uniform_(
        max(0.0, 1.0 - saturation_jitter),
        1.0 + saturation_jitter,
    )

    grayscale = (0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]).repeat(1, 3, 1, 1)
    jittered = image * brightness_scale
    jittered = (jittered - jittered.mean(dim=(2, 3), keepdim=True)) * contrast_scale + jittered.mean(
        dim=(2, 3), keepdim=True
    )
    jittered = grayscale + (jittered - grayscale) * saturation_scale

    if hue_jitter > 0.0:
        hue_mix = torch.empty(image.shape[0], 1, 1, 1, device=image.device, dtype=image.dtype).uniform_(
            -hue_jitter,
            hue_jitter,
        )
        channel_mean = jittered.mean(dim=1, keepdim=True)
        jittered = jittered + hue_mix * torch.cat(
            (
                jittered[:, 1:2] - channel_mean,
                jittered[:, 2:3] - channel_mean,
                jittered[:, 0:1] - channel_mean,
            ),
            dim=1,
        )

    return jittered.clamp(0.0, 1.0)


def apply_gaussian_noise(image: Tensor, std: float) -> Tensor:
    """Apply additive Gaussian noise."""

    return (image + torch.randn_like(image) * std).clamp(0.0, 1.0)


class RandomAugmentationPipeline(nn.Module):
    """Compose a random subset of differentiable robustness attacks."""

    def __init__(self) -> None:
        super().__init__()
        self.config = DEFAULT_CONFIG.augmentations

    def forward(self, image: Tensor) -> Tensor:
        """Apply a random subset of configured attacks to an image batch."""

        attack_specs: list[tuple[str, float, AttackFn]] = [
            (
                "jpeg",
                self.config.jpeg_probability,
                lambda tensor: apply_differentiable_jpeg(
                    tensor,
                    quality=int(
                        torch.randint(
                            self.config.jpeg_quality_min,
                            self.config.jpeg_quality_max + 1,
                            (1,),
                            device=tensor.device,
                        ).item()
                    ),
                ),
            ),
            (
                "blur",
                self.config.blur_probability,
                lambda tensor: apply_gaussian_blur(
                    tensor,
                    kernel_size=self.config.blur_kernel_size,
                    sigma=float(
                        torch.empty(1, device=tensor.device).uniform_(
                            self.config.blur_sigma_min,
                            self.config.blur_sigma_max,
                        )
                    ),
                ),
            ),
            (
                "crop",
                self.config.crop_probability,
                lambda tensor: apply_random_crop_resize(
                    tensor,
                    scale_min=self.config.crop_scale_min,
                    scale_max=self.config.crop_scale_max,
                ),
            ),
            (
                "color",
                self.config.color_jitter_probability,
                lambda tensor: apply_color_jitter(
                    tensor,
                    brightness_jitter=self.config.brightness_jitter,
                    contrast_jitter=self.config.contrast_jitter,
                    saturation_jitter=self.config.saturation_jitter,
                    hue_jitter=self.config.hue_jitter,
                ),
            ),
            (
                "noise",
                self.config.noise_probability,
                lambda tensor: apply_gaussian_noise(
                    tensor,
                    std=float(
                        torch.empty(1, device=tensor.device).uniform_(
                            self.config.noise_std_min,
                            self.config.noise_std_max,
                        )
                    ),
                ),
            ),
        ]

        chosen_attacks = [spec for spec in attack_specs if torch.rand(1, device=image.device).item() < spec[1]]
        if not chosen_attacks:
            fallback_index = int(torch.randint(0, len(attack_specs), (1,), device=image.device).item())
            chosen_attacks = [attack_specs[fallback_index]]

        max_attacks = min(self.config.max_active_attacks, len(chosen_attacks))
        min_attacks = min(self.config.min_active_attacks, max_attacks)
        attack_count = int(torch.randint(min_attacks, max_attacks + 1, (1,), device=image.device).item())
        permutation = torch.randperm(len(chosen_attacks), device=image.device)

        attacked = image
        for index in permutation[:attack_count]:
            attacked = chosen_attacks[int(index.item())][2](attacked)
        return attacked.clamp(0.0, 1.0)


def build_evaluation_attack_suite() -> dict[str, AttackFn]:
    """Build a fixed-parameter attack suite for evaluation."""

    config = DEFAULT_CONFIG.evaluation
    augmentations = DEFAULT_CONFIG.augmentations
    return {
        "clean": lambda image: image,
        "jpeg": lambda image: apply_differentiable_jpeg(image, quality=config.jpeg_quality),
        "blur": lambda image: apply_gaussian_blur(
            image,
            kernel_size=augmentations.blur_kernel_size,
            sigma=config.blur_sigma,
        ),
        "crop": lambda image: apply_random_crop_resize(
            image,
            scale_min=config.crop_scale,
            scale_max=config.crop_scale,
        ),
        "noise": lambda image: apply_gaussian_noise(image, std=config.noise_std),
        "brightness": lambda image: (image * config.brightness_factor).clamp(0.0, 1.0),
    }


def _run_smoke_test() -> None:
    """Run a lightweight local validation of the augmentation pipeline."""

    torch.manual_seed(0)
    pipeline = RandomAugmentationPipeline()
    images = torch.rand(2, 3, DEFAULT_CONFIG.data.image_size, DEFAULT_CONFIG.data.image_size)
    attacked = pipeline(images)
    assert attacked.shape == images.shape
    assert torch.all(attacked >= 0.0) and torch.all(attacked <= 1.0)
    print("Augmentation smoke test passed.")


if __name__ == "__main__":
    _run_smoke_test()
