"""Image quality and watermark detection metrics for PyWatermark."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def bit_accuracy_from_probs(predicted_bits: Tensor, target_bits: Tensor, threshold: float = 0.5) -> Tensor:
    """Compute mean per-bit accuracy from sigmoid probabilities."""

    predictions = (predicted_bits >= threshold).to(target_bits.dtype)
    return (predictions == target_bits).to(torch.float32).mean()


def exact_match_accuracy_from_probs(
    predicted_bits: Tensor,
    target_bits: Tensor,
    threshold: float = 0.5,
) -> Tensor:
    """Compute the fraction of samples whose full bit vectors match exactly."""

    predictions = (predicted_bits >= threshold).to(target_bits.dtype)
    return (predictions == target_bits).all(dim=1).to(torch.float32).mean()


def peak_signal_to_noise_ratio(reference: Tensor, estimate: Tensor, max_value: float = 1.0) -> Tensor:
    """Compute average PSNR in decibels for a batch of normalized images."""

    mse = F.mse_loss(estimate, reference, reduction="none").flatten(1).mean(dim=1)
    return 10.0 * torch.log10(torch.tensor(max_value**2, device=reference.device) / mse.clamp_min(1e-8)).mean()


def _gaussian_kernel(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Build a normalized 1D Gaussian kernel."""

    offsets = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    kernel = torch.exp(-(offsets**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def _ssim_window(channels: int, window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Build a depthwise 2D Gaussian SSIM window."""

    kernel_1d = _gaussian_kernel(window_size, sigma, device=device, dtype=dtype)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    window = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def structural_similarity(
    reference: Tensor,
    estimate: Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    max_value: float = 1.0,
) -> Tensor:
    """Compute average SSIM for a batch of normalized images."""

    if reference.shape != estimate.shape:
        raise ValueError(
            f"reference and estimate must share shape; received {tuple(reference.shape)} and {tuple(estimate.shape)}."
        )
    if reference.ndim != 4:
        raise ValueError(f"Expected 4D image tensors, received {tuple(reference.shape)}.")

    channels = reference.shape[1]
    window = _ssim_window(channels, window_size, sigma, reference.device, reference.dtype)
    padding = window_size // 2

    mu_reference = F.conv2d(reference, window, padding=padding, groups=channels)
    mu_estimate = F.conv2d(estimate, window, padding=padding, groups=channels)

    mu_reference_sq = mu_reference.pow(2)
    mu_estimate_sq = mu_estimate.pow(2)
    mu_reference_estimate = mu_reference * mu_estimate

    sigma_reference_sq = F.conv2d(reference * reference, window, padding=padding, groups=channels) - mu_reference_sq
    sigma_estimate_sq = F.conv2d(estimate * estimate, window, padding=padding, groups=channels) - mu_estimate_sq
    sigma_reference_estimate = (
        F.conv2d(reference * estimate, window, padding=padding, groups=channels) - mu_reference_estimate
    )

    c1 = (0.01 * max_value) ** 2
    c2 = (0.03 * max_value) ** 2

    numerator = (2 * mu_reference_estimate + c1) * (2 * sigma_reference_estimate + c2)
    denominator = (mu_reference_sq + mu_estimate_sq + c1) * (sigma_reference_sq + sigma_estimate_sq + c2)
    ssim_map = numerator / denominator.clamp_min(1e-8)
    return ssim_map.mean()


def format_metric(value: float | Tensor, precision: int = 4) -> str:
    """Format a scalar metric consistently for logs and reports."""

    scalar = float(value.item()) if isinstance(value, Tensor) else float(value)
    return f"{scalar:.{precision}f}"


def agreement_entropy(predicted_bits: Tensor) -> Tensor:
    """Estimate the entropy of decoder bit probabilities."""

    probabilities = predicted_bits.clamp(1e-6, 1.0 - 1e-6)
    entropy = -(probabilities * torch.log2(probabilities) + (1.0 - probabilities) * torch.log2(1.0 - probabilities))
    return entropy.mean()


def binary_string_to_tensor(bits: str) -> Tensor:
    """Convert a binary string such as ``0101`` into a float tensor."""

    if any(bit not in {"0", "1"} for bit in bits):
        raise ValueError("Bit strings must contain only '0' and '1'.")
    values = [float(bit) for bit in bits]
    return torch.tensor(values, dtype=torch.float32)


def tensor_to_binary_string(bits: Tensor, threshold: float = 0.5) -> str:
    """Convert a probability or binary tensor into a compact bit string."""

    flattened = bits.detach().view(-1)
    binary = (flattened >= threshold).to(torch.int64).cpu().tolist()
    return "".join(str(bit) for bit in binary)
