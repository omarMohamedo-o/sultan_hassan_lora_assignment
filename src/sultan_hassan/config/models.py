"""Configuration schema validated with Pydantic v2."""

from pathlib import Path

from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    """General project configuration."""

    name: str = "sultan-hassan-flux"
    version: str = "0.1.0"
    random_seed: int = 42
    device: str = "cuda"  # "cuda", "cpu", or "auto"


class PathsConfig(BaseModel):
    """File system directory locations."""

    raw_videos: Path = Path("data/raw/videos")
    raw_external: Path = Path("data/raw/external")
    frames_raw: Path = Path("data/frames/raw")
    frames_quality_filtered: Path = Path("data/frames/quality_filtered")
    frames_deduplicated: Path = Path("data/frames/deduplicated")
    candidates_images: Path = Path("data/candidates/images")
    candidates_metadata: Path = Path("data/candidates/metadata")
    review_contact_sheets: Path = Path("data/review/contact_sheets")
    review_decisions: Path = Path("data/review/decisions")
    final_images: Path = Path("data/final/images")
    final_captions: Path = Path("data/final/captions")
    dataset_splits: Path = Path("data/final/splits")
    metadata_dir: Path = Path("data/metadata")
    models_lora: Path = Path("models/lora")
    outputs_tests: Path = Path("outputs/tests")
    outputs_evaluation: Path = Path("outputs/evaluation")
    reports_contact_sheets: Path = Path("reports/contact_sheets")
    reports_dataset: Path = Path("reports/dataset")
    reports_evaluation: Path = Path("reports/evaluation")
    deliverable: Path = Path("deliverable")


class VideoConfig(BaseModel):
    """Video ingestion and sampling configuration."""

    sampling_mode: str = "seconds"  # "seconds" or "fps"
    sample_interval_seconds: float = 0.5
    fps_rate: float = 2.0
    supported_extensions: list[str] = Field(
        default_factory=lambda: [".mp4", ".mov", ".avi", ".mkv"]
    )


class FramingConfig(BaseModel):
    """Image framing, bucketing, and crop configuration for training."""

    mode: str = "bucket"  # "bucket", "smart_center_crop", "letterbox"
    target_width: int = 1024
    target_height: int = 1024
    min_short_side: int = 1024
    aspect_ratio_buckets: list[list[int]] = Field(
        default_factory=lambda: [
            [1024, 1024],
            [896, 1152],
            [1152, 896],
            [768, 1344],
            [1344, 768],
        ]
    )
    pad_color: list[int] = Field(default_factory=lambda: [0, 0, 0])


class ImageConfig(BaseModel):
    """Image resolution and geometry configuration."""

    min_short_side: int = 1024
    target_size: int = 1024
    preserve_aspect_ratio: bool = True
    framing: FramingConfig = Field(default_factory=FramingConfig)


class QualityConfig(BaseModel):
    """Automated quality metric thresholds."""

    blur_threshold: float = 100.0
    brightness_min: float = 30.0
    brightness_max: float = 235.0
    contrast_min: float = 25.0
    max_aspect_ratio: float = 2.5
    min_raw_short_side: int = 512


class DeduplicationConfig(BaseModel):
    """Perceptual hashing deduplication thresholds."""

    method: str = "phash"  # "phash" or "dhash"
    hash_size: int = 8
    distance_threshold: int = 6


class SemanticConfig(BaseModel):
    """Sultan Hassan vs. Al-Rifa'i filtering parameters."""

    enabled: bool = True
    rifai_keywords: list[str] = Field(
        default_factory=lambda: ["الرفاعي", "رفاعي", "rifai", "al-rifai", "al rifai"]
    )
    sultan_hassan_keywords: list[str] = Field(
        default_factory=lambda: [
            "سلطان حسن",
            "السلطان حسن",
            "sultan hassan",
            "sultan hasan",
            "sltnhsn",
        ]
    )
    clip_enabled: bool = False
    clip_model: str = "ViT-B/32"
    rifai_review_threshold: float = 0.15


class SplitConfig(BaseModel):
    """Train / Test / Validation dataset split configuration."""

    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    fixed_train_count: int | None = None
    fixed_val_count: int | None = None
    fixed_test_count: int | None = None
    shuffle: bool = True
    seed: int = 42


class DatasetConfig(BaseModel):
    """Final dataset targets and constraints."""

    min_images: int = 25
    max_images: int = 40
    trigger_word: str = "sltnhsn"
    split: SplitConfig = Field(default_factory=SplitConfig)
    target_distribution: dict[str, list[int]] = Field(
        default_factory=lambda: {
            "exterior_facade": [7, 8],
            "entrance": [4, 5],
            "courtyard": [5, 6],
            "iwan": [4, 5],
            "minaret": [3, 4],
            "dome": [2, 3],
            "interior_ornament": [4, 5],
        }
    )


class TrainingConfig(BaseModel):
    """FLUX LoRA hyperparameters and batching."""

    trigger_word: str = "sltnhsn"
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    lora_rank: int = 16
    lora_alpha: int = 16
    learning_rate: float = 1.0e-4
    optimizer: str = "AdamW8bit"
    scheduler: str = "constant"
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int = 2
    max_train_steps: int = 1000
    epochs: int = 10
    seed: int = 42
    precision: str = "bf16"
    checkpoint_frequency: int = 250


class InferenceConfig(BaseModel):
    """Inference runtime configuration.

    Supports multiple backends:
    - "onnxruntime": Python ONNX Runtime (default, lightweight)
    - "pytorch": Standard PyTorch/Diffusers pipeline
    - "cpp_onnx": C++ ONNX Runtime binary (maximum performance)

    Execution providers (for onnxruntime backend), tried in order:
    - "DmlExecutionProvider": DirectML (Windows GPU – AMD/Intel/NVIDIA)
    - "CUDAExecutionProvider": NVIDIA CUDA
    - "TensorrtExecutionProvider": NVIDIA TensorRT (fastest NVIDIA)
    - "OpenVINOExecutionProvider": Intel OpenVINO
    - "CPUExecutionProvider": CPU fallback (always available)
    """

    backend: str = "onnxruntime"  # "onnxruntime", "pytorch", "cpp_onnx"
    execution_providers: list[str] = Field(
        default_factory=lambda: [
            "DmlExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    )
    onnx_model_path: str = "models/onnx/sltnhsn_flux_lora.onnx"
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    width: int = 1024
    height: int = 1024
    seeds: list[int] = Field(default_factory=lambda: [101, 202, 303, 404])


class AppConfig(BaseModel):
    """Root application configuration."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    semantic: SemanticConfig = Field(default_factory=SemanticConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)

