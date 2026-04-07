"""Device selection helpers for PyWatermark."""

from __future__ import annotations

import torch


def get_best_device() -> tuple[torch.device, str]:
    """Return the preferred torch device and a readable description."""

    if torch.cuda.is_available():
        cuda_device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        return cuda_device, f"CUDA ({device_name})"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return torch.device("mps"), "MPS"

    return torch.device("cpu"), "CPU"
