"""Random seeding helpers for reproducible PyWatermark runs."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic_algorithms: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic_algorithms)


def seed_worker(worker_id: int) -> None:
    """Seed a dataloader worker from the initial PyTorch worker seed."""

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)
