"""Pydantic domain models for data contracts across the pipeline with strict type validation."""

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from sultan_hassan.domain.enums import (
    CategoryEnum,
    PipelineStage,
    QualityStatus,
    ReviewDecisionEnum,
    SemanticLabel,
)


class VideoMetadata(BaseModel):
    """Metadata extracted during video inspection with strict integer validation."""

    filename: str
    path: str
    sha256: str
    file_size_bytes: int = Field(ge=0, description="File size in bytes")
    duration_seconds: float = Field(ge=0.0, description="Duration in seconds")
    width: int = Field(gt=0, description="Frame width in pixels")
    height: int = Field(gt=0, description="Frame height in pixels")
    fps: float = Field(gt=0.0, description="Frames per second")
    frame_count: int = Field(ge=0, description="Total frame count")
    codec: str = "unknown"
    container: str = "unknown"
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class FrameMetadata(BaseModel):
    """Metadata recorded for each extracted video frame with strict integer validation."""

    frame_id: str
    video_id: str
    frame_number: int = Field(ge=0, description="Zero-indexed frame position")
    timestamp_seconds: float = Field(ge=0.0, description="Timestamp from start of video")
    width: int = Field(gt=0, description="Width in pixels")
    height: int = Field(gt=0, description="Height in pixels")
    source_video: str
    source_sha256: str
    local_filename: str
    sha256: str = ""
    stage: PipelineStage = PipelineStage.EXTRACTED


class ImageQualityResult(BaseModel):
    """Detailed quality assessment for a frame with strict integer validation."""

    frame_id: str
    local_filename: str
    width: int = Field(ge=0, description="Width in pixels")
    height: int = Field(ge=0, description="Height in pixels")
    short_side: int = Field(ge=0, description="Shortest dimension in pixels")
    blur_score: float = Field(ge=0.0, description="Laplacian variance")
    brightness: float = Field(ge=0.0, le=255.0, description="Mean intensity")
    contrast: float = Field(ge=0.0, description="Standard deviation of intensity")
    aspect_ratio: float = Field(ge=0.0, description="Long side divided by short side")
    quality_status: QualityStatus
    rejection_reason: str = ""


class DuplicateCluster(BaseModel):
    """Cluster of near-duplicate frames with a selected representative."""

    cluster_id: str
    representative_frame_id: str
    frame_ids: list[str] = Field(default_factory=list)
    hash_value: str
    size: int = Field(ge=1, description="Cluster membership size")


class SemanticFilterResult(BaseModel):
    """Multi-signal classification distinguishing Sultan Hassan from Al-Rifa'i."""

    frame_id: str
    local_filename: str
    sultan_hassan_score: float = Field(ge=0.0, le=1.0)
    al_rifai_score: float = Field(ge=0.0, le=1.0)
    score_difference: float
    semantic_label: SemanticLabel
    visual_dominant: str
    status: QualityStatus
    rationale: str = ""


class ReviewDecision(BaseModel):
    """Human decision recorded during candidate review."""

    frame_id: str
    decision: ReviewDecisionEnum
    category: CategoryEnum = CategoryEnum.OTHER
    reviewer: str = "human"
    notes: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class FinalImageRecord(BaseModel):
    """Provenance and specification metadata for final dataset images."""

    image_id: str
    source_video: str
    source_sha256: str = ""
    frame_number: int = Field(ge=0, description="Original source frame number")
    timestamp_seconds: float = Field(ge=0.0, description="Original timestamp in seconds")
    original_width: int = Field(gt=0, description="Original frame width in pixels")
    original_height: int = Field(gt=0, description="Original frame height in pixels")
    processed_width: int = Field(gt=0, description="Final training width in pixels")
    processed_height: int = Field(gt=0, description="Final training height in pixels")
    file_path: str
    sha256: str
    phash: str
    category: CategoryEnum
    review_decision: ReviewDecisionEnum
    sultan_hassan_score: float = 1.0
    al_rifai_score: float = 0.0
    source_type: str = "user_provided_video"
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class CaptionRecord(BaseModel):
    """Validation record for an image caption with integer length metrics."""

    image_id: str
    image_filename: str
    caption: str
    trigger_word_present: bool = True
    char_length: int = Field(ge=0, description="Total characters")
    word_count: int = Field(ge=0, description="Total words")


class DatasetValidationResult(BaseModel):
    """Comprehensive validation result of final dataset with validated integer counters."""

    is_valid: bool
    total_images: int = Field(ge=0, description="Total image count")
    error_count: int = Field(ge=0, description="Validation error count")
    warning_count: int = Field(ge=0, description="Validation warning count")
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class PipelineItemState(BaseModel):
    """Lifecycle tracking for an individual artifact in the pipeline."""

    item_id: str
    stage: PipelineStage
    file_path: str
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class PipelineManifest(BaseModel):
    """Persistent pipeline state allowing fully resumable workflows."""

    project_name: str = "sultan-hassan-flux"
    items: dict[str, PipelineItemState] = Field(default_factory=dict)
    last_updated: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def mark_stage(self, item_id: str, stage: PipelineStage, file_path: str | Path) -> None:
        """Update or insert item stage."""
        self.items[item_id] = PipelineItemState(
            item_id=item_id,
            stage=stage,
            file_path=str(file_path),
            updated_at=datetime.now(UTC).isoformat(),
        )
        self.last_updated = datetime.now(UTC).isoformat()

    def get_stage(self, item_id: str) -> PipelineStage | None:
        """Retrieve current stage of item."""
        if item_id in self.items:
            return self.items[item_id].stage
        return None
