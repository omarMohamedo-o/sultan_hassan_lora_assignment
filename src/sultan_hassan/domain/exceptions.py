"""Custom domain exceptions for Sultan Hassan ML Pipeline."""


class SultanHassanError(Exception):
    """Base exception for all domain errors."""


# Backward compatibility alias
SultanHassanException = SultanHassanError


class ConfigurationError(SultanHassanError):
    """Raised when configuration validation or loading fails."""


class VideoInspectionError(SultanHassanError):
    """Raised when inspecting a video file fails."""


class FrameExtractionError(SultanHassanError):
    """Raised during video frame extraction."""


class QualityFilterError(SultanHassanError):
    """Raised during image quality assessment."""


class DeduplicationError(SultanHassanError):
    """Raised during duplicate clustering and hashing."""


class DatasetValidationError(SultanHassanError):
    """Raised when final dataset validation fails critical quality gates."""
