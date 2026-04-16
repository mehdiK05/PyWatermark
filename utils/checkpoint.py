"""Checkpoint helpers for PyWatermark training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return its resolved path."""

    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_checkpoint(state: dict[str, Any], path: str | Path) -> Path:
    """Persist a training checkpoint to disk."""

    checkpoint_path = Path(path).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_path)
    return checkpoint_path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint dictionary from disk."""

    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint at {checkpoint_path} did not contain a dictionary payload.")
    return checkpoint


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Return the most recently modified checkpoint file in a directory."""

    directory = Path(checkpoint_dir).expanduser()
    if not directory.exists():
        return None

    candidates = sorted(directory.glob("*.pt"), key=lambda candidate: candidate.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None
