"""Central configuration values for the PyWatermark project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_NAME = "Mimicry"
PROJECT_DESCRIPTION = "A SynthID-inspired invisible image watermarking system in PyTorch"

DEFAULT_IMAGE_SIZE = 128
DEFAULT_IMAGE_CHANNELS = 3
DEFAULT_KEY_BITS = 48

DEFAULT_TRAIN_BATCH_SIZE = 16
DEFAULT_EVAL_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 2

DEFAULT_RANDOM_SEED = 42
DEFAULT_ENCODER_ALPHA = 0.05
DEFAULT_ENCODER_BASE_CHANNELS = 48
DEFAULT_DECODER_BASE_CHANNELS = 48
DEFAULT_DECODER_BLOCKS = 4

DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_GRAD_CLIP_NORM = 1.0
DEFAULT_LR_PATIENCE = 2
DEFAULT_LR_FACTOR = 0.5
DEFAULT_NUM_EPOCHS = 10
DEFAULT_LOG_INTERVAL = 10
DEFAULT_IMAGE_LOG_INTERVAL = 100
DEFAULT_CHECKPOINT_INTERVAL = 1
DEFAULT_MAX_BATCH_SIZE_RETRIES = 5

DEFAULT_INVISIBILITY_WEIGHT = 1.0
DEFAULT_DETECTION_WEIGHT = 4.0
DEFAULT_SSIM_WEIGHT = 1.0
DEFAULT_L2_WEIGHT = 1.0

DEFAULT_THRESHOLD = 0.5
DEFAULT_REPORT_FILENAME = "evaluation_report.txt"

DEFAULT_JPEG_PROBABILITY = 0.75
DEFAULT_BLUR_PROBABILITY = 0.60
DEFAULT_CROP_PROBABILITY = 0.60
DEFAULT_COLOR_JITTER_PROBABILITY = 0.60
DEFAULT_NOISE_PROBABILITY = 0.60
DEFAULT_MIN_ACTIVE_ATTACKS = 1
DEFAULT_MAX_ACTIVE_ATTACKS = 3
DEFAULT_JPEG_QUALITY_MIN = 45
DEFAULT_JPEG_QUALITY_MAX = 95
DEFAULT_BLUR_KERNEL_SIZE = 5
DEFAULT_BLUR_SIGMA_MIN = 0.1
DEFAULT_BLUR_SIGMA_MAX = 1.5
DEFAULT_CROP_SCALE_MIN = 0.70
DEFAULT_CROP_SCALE_MAX = 1.00
DEFAULT_BRIGHTNESS_JITTER = 0.20
DEFAULT_CONTRAST_JITTER = 0.20
DEFAULT_SATURATION_JITTER = 0.15
DEFAULT_HUE_JITTER = 0.03
DEFAULT_NOISE_STD_MIN = 0.0
DEFAULT_NOISE_STD_MAX = 0.03
DEFAULT_ATTACKS_ENABLED = True

EVAL_JPEG_QUALITY = 55
EVAL_BLUR_SIGMA = 1.25
EVAL_CROP_SCALE = 0.80
EVAL_NOISE_STD = 0.03
EVAL_BRIGHTNESS_FACTOR = 1.15

TRAIN_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

DEFAULT_DATA_ROOT = Path("datasets")
DEFAULT_TRAIN_DIR = DEFAULT_DATA_ROOT / "train"
DEFAULT_VAL_DIR = DEFAULT_DATA_ROOT / "val"
DEFAULT_TEST_DIR = DEFAULT_DATA_ROOT / "test"

DEFAULT_ARTIFACT_ROOT = Path("artifacts")
DEFAULT_CHECKPOINT_DIR = DEFAULT_ARTIFACT_ROOT / "checkpoints"
DEFAULT_LOG_DIR = DEFAULT_ARTIFACT_ROOT / "logs"
DEFAULT_REPORT_DIR = DEFAULT_ARTIFACT_ROOT / "reports"
DEFAULT_OUTPUT_DIR = DEFAULT_ARTIFACT_ROOT / "outputs"
DEFAULT_RAW_DATA_DIR = Path("datasets_raw")

DEFAULT_COLAB_ROOT = Path("/content")
DEFAULT_COLAB_PROJECT_DIR = DEFAULT_COLAB_ROOT / "PyWatermark"
DEFAULT_COLAB_DRIVE_DIR = DEFAULT_COLAB_ROOT / "drive" / "MyDrive" / "PyWatermark"

DEFAULT_COCO_DOWNLOAD_SPLIT = "val2017"
DEFAULT_COCO_TRAIN_COUNT = 4000
DEFAULT_COCO_VAL_COUNT = 500
DEFAULT_COCO_TEST_COUNT = 500
COCO_IMAGE_URLS: dict[str, str] = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
}


@dataclass(frozen=True)
class PathConfig:
    """Filesystem defaults used across training, evaluation, and demos."""

    data_root: Path = DEFAULT_DATA_ROOT
    train_dir: Path = DEFAULT_TRAIN_DIR
    val_dir: Path = DEFAULT_VAL_DIR
    test_dir: Path = DEFAULT_TEST_DIR
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    log_dir: Path = DEFAULT_LOG_DIR
    report_dir: Path = DEFAULT_REPORT_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    raw_data_dir: Path = DEFAULT_RAW_DATA_DIR
    colab_root: Path = DEFAULT_COLAB_ROOT
    colab_project_dir: Path = DEFAULT_COLAB_PROJECT_DIR
    colab_drive_dir: Path = DEFAULT_COLAB_DRIVE_DIR


@dataclass(frozen=True)
class DataConfig:
    """Dataset and dataloader defaults."""

    image_size: int = DEFAULT_IMAGE_SIZE
    image_channels: int = DEFAULT_IMAGE_CHANNELS
    key_bits: int = DEFAULT_KEY_BITS
    train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE
    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS
    train_shuffle: bool = True
    drop_last_train: bool = True
    deterministic_eval_keys: bool = True
    eval_key_seed: int = DEFAULT_RANDOM_SEED + 1
    extensions: tuple[str, ...] = field(default_factory=lambda: TRAIN_EXTENSIONS)


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime preferences shared by scripts."""

    random_seed: int = DEFAULT_RANDOM_SEED
    prefer_cuda: bool = True
    prefer_mps: bool = True
    cudnn_benchmark: bool = True
    deterministic_algorithms: bool = False


@dataclass(frozen=True)
class EncoderConfig:
    """Defaults for the watermark encoder."""

    image_channels: int = DEFAULT_IMAGE_CHANNELS
    base_channels: int = DEFAULT_ENCODER_BASE_CHANNELS
    alpha: float = DEFAULT_ENCODER_ALPHA


@dataclass(frozen=True)
class DecoderConfig:
    """Defaults for the watermark decoder."""

    image_channels: int = DEFAULT_IMAGE_CHANNELS
    base_channels: int = DEFAULT_DECODER_BASE_CHANNELS
    residual_blocks: int = DEFAULT_DECODER_BLOCKS


@dataclass(frozen=True)
class AugmentationConfig:
    """Defaults for differentiable robustness attacks used during training."""

    enabled: bool = DEFAULT_ATTACKS_ENABLED
    jpeg_probability: float = DEFAULT_JPEG_PROBABILITY
    blur_probability: float = DEFAULT_BLUR_PROBABILITY
    crop_probability: float = DEFAULT_CROP_PROBABILITY
    color_jitter_probability: float = DEFAULT_COLOR_JITTER_PROBABILITY
    noise_probability: float = DEFAULT_NOISE_PROBABILITY
    min_active_attacks: int = DEFAULT_MIN_ACTIVE_ATTACKS
    max_active_attacks: int = DEFAULT_MAX_ACTIVE_ATTACKS
    jpeg_quality_min: int = DEFAULT_JPEG_QUALITY_MIN
    jpeg_quality_max: int = DEFAULT_JPEG_QUALITY_MAX
    blur_kernel_size: int = DEFAULT_BLUR_KERNEL_SIZE
    blur_sigma_min: float = DEFAULT_BLUR_SIGMA_MIN
    blur_sigma_max: float = DEFAULT_BLUR_SIGMA_MAX
    crop_scale_min: float = DEFAULT_CROP_SCALE_MIN
    crop_scale_max: float = DEFAULT_CROP_SCALE_MAX
    brightness_jitter: float = DEFAULT_BRIGHTNESS_JITTER
    contrast_jitter: float = DEFAULT_CONTRAST_JITTER
    saturation_jitter: float = DEFAULT_SATURATION_JITTER
    hue_jitter: float = DEFAULT_HUE_JITTER
    noise_std_min: float = DEFAULT_NOISE_STD_MIN
    noise_std_max: float = DEFAULT_NOISE_STD_MAX


@dataclass(frozen=True)
class LossConfig:
    """Loss weighting defaults."""

    invisibility_weight: float = DEFAULT_INVISIBILITY_WEIGHT
    detection_weight: float = DEFAULT_DETECTION_WEIGHT
    ssim_weight: float = DEFAULT_SSIM_WEIGHT
    l2_weight: float = DEFAULT_L2_WEIGHT


@dataclass(frozen=True)
class TrainingConfig:
    """Optimizer, scheduling, and logging defaults."""

    epochs: int = DEFAULT_NUM_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    grad_clip_norm: float = DEFAULT_GRAD_CLIP_NORM
    lr_patience: int = DEFAULT_LR_PATIENCE
    lr_factor: float = DEFAULT_LR_FACTOR
    log_interval: int = DEFAULT_LOG_INTERVAL
    image_log_interval: int = DEFAULT_IMAGE_LOG_INTERVAL
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL
    use_amp: bool = True
    max_batch_size_retries: int = DEFAULT_MAX_BATCH_SIZE_RETRIES
    tensorboard_image_count: int = 4


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation defaults and fixed robustness attack parameters."""

    threshold: float = DEFAULT_THRESHOLD
    report_filename: str = DEFAULT_REPORT_FILENAME
    jpeg_quality: int = EVAL_JPEG_QUALITY
    blur_sigma: float = EVAL_BLUR_SIGMA
    crop_scale: float = EVAL_CROP_SCALE
    noise_std: float = EVAL_NOISE_STD
    brightness_factor: float = EVAL_BRIGHTNESS_FACTOR


@dataclass(frozen=True)
class DemoConfig:
    """Interactive demo defaults."""

    share: bool = True
    server_name: str = "0.0.0.0"
    server_port: int = 7860


@dataclass(frozen=True)
class DatasetPrepConfig:
    """Defaults for dataset preparation utilities."""

    coco_download_split: str = DEFAULT_COCO_DOWNLOAD_SPLIT
    train_count: int = DEFAULT_COCO_TRAIN_COUNT
    val_count: int = DEFAULT_COCO_VAL_COUNT
    test_count: int = DEFAULT_COCO_TEST_COUNT
    coco_image_urls: dict[str, str] = field(default_factory=lambda: dict(COCO_IMAGE_URLS))


@dataclass(frozen=True)
class ProjectConfig:
    """Grouped project configuration."""

    project_name: str = PROJECT_NAME
    project_description: str = PROJECT_DESCRIPTION
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    augmentations: AugmentationConfig = field(default_factory=AugmentationConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)
    dataset_prep: DatasetPrepConfig = field(default_factory=DatasetPrepConfig)


DEFAULT_CONFIG = ProjectConfig()
