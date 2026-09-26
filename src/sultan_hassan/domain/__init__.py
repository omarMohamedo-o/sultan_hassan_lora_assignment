"""Domain models and enumerations."""

from sultan_hassan.domain.enums import (
    CategoryEnum,
    PipelineStage,
    QualityStatus,
    ReviewDecisionEnum,
    SemanticLabel,
)
from sultan_hassan.domain.exceptions import (
    ConfigurationError,
    DatasetValidationError,
    DeduplicationError,
    FrameExtractionError,
    QualityFilterError,
    SultanHassanException,
    VideoInspectionError,
)
from sultan_hassan.domain.models import (
    CaptionRecord,
    DatasetValidationResult,
    DuplicateCluster,
    FinalImageRecord,
    FrameMetadata,
    ImageQualityResult,
    PipelineItemState,
    PipelineManifest,
    ReviewDecision,
    SemanticFilterResult,
    VideoMetadata,
)

__all__ = [
    "CategoryEnum",
    "PipelineStage",
    "QualityStatus",
    "ReviewDecisionEnum",
    "SemanticLabel",
    "SultanHassanException",
    "ConfigurationError",
    "VideoInspectionError",
    "FrameExtractionError",
    "QualityFilterError",
    "DeduplicationError",
    "DatasetValidationError",
    "VideoMetadata",
    "FrameMetadata",
    "ImageQualityResult",
    "DuplicateCluster",
    "SemanticFilterResult",
    "ReviewDecision",
    "FinalImageRecord",
    "CaptionRecord",
    "DatasetValidationResult",
    "PipelineItemState",
    "PipelineManifest",
]
