"""Central configuration values for the PyWatermark project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_NAME = "PyWatermark"
DEFAULT_IMAGE_SIZE = 128
DEFAULT_KEY_BITS = 48
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 4
DEFAULT_SHUFFLE = True
DEFAULT_NUM_BATCHES_PREVIEW = 2
TRAIN_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
DEFAULT_DATA_DIR = Path("datasets")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_LOG_DIR = Path("logs")
RANDOM_SEED = 42


@dataclass(frozen=True)
class PathConfig:
    """Filesystem defaults used across the project."""

    data_dir: Path = DEFAULT_DATA_DIR
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    log_dir: Path = DEFAULT_LOG_DIR


@dataclass(frozen=True)
class DataConfig:
    """Dataset and dataloader defaults for Phase 1."""

    image_size: int = DEFAULT_IMAGE_SIZE
    key_bits: int = DEFAULT_KEY_BITS
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS
    shuffle: bool = DEFAULT_SHUFFLE
    num_batches_preview: int = DEFAULT_NUM_BATCHES_PREVIEW
    extensions: tuple[str, ...] = field(default_factory=lambda: tuple(TRAIN_EXTENSIONS))
    drop_last: bool = False
    pin_memory: bool | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime preferences shared by scripts."""

    random_seed: int = RANDOM_SEED
    prefer_cuda: bool = True
    prefer_mps: bool = True


@dataclass(frozen=True)
class ProjectConfig:
    """Grouped project configuration."""

    project_name: str = PROJECT_NAME
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


DEFAULT_CONFIG = ProjectConfig()
