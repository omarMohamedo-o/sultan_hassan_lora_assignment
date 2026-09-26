"""Domain enumerations for the Sultan Hassan dataset pipeline."""

from enum import StrEnum


class QualityStatus(StrEnum):
    """Status assigned by automated image quality filter."""

    KEEP = "KEEP"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


class SemanticLabel(StrEnum):
    """Architectural classification label."""

    SULTAN_HASSAN = "SULTAN_HASSAN"
    AL_RIFAI = "AL_RIFAI"
    MIXED = "MIXED"
    UNCERTAIN = "UNCERTAIN"
    OTHER = "OTHER"


class ReviewDecisionEnum(StrEnum):
    """Decisions available during human review."""

    KEEP = "KEEP"
    REJECT = "REJECT"
    AL_RIFAI = "AL_RIFAI"
    MIXED = "MIXED"
    WRONG_BUILDING = "WRONG_BUILDING"
    PEOPLE = "PEOPLE"
    WATERMARK = "WATERMARK"
    LOW_QUALITY = "LOW_QUALITY"
    DUPLICATE = "DUPLICATE"
    UNCERTAIN = "UNCERTAIN"


class CategoryEnum(StrEnum):
    """Architectural category for diversity coverage."""

    EXTERIOR = "EXTERIOR"
    ENTRANCE = "ENTRANCE"
    COURTYARD = "COURTYARD"
    IWAN = "IWAN"
    MINARET = "MINARET"
    DOME = "DOME"
    INTERIOR = "INTERIOR"
    MUQARNAS = "MUQARNAS"
    ORNAMENT = "ORNAMENT"
    HANGING_LAMP = "HANGING_LAMP"
    OTHER = "OTHER"


class PipelineStage(StrEnum):
    """Lifecycle stage of a frame/dataset item."""

    DISCOVERED = "DISCOVERED"
    EXTRACTED = "EXTRACTED"
    QUALITY_CHECKED = "QUALITY_CHECKED"
    DEDUPLICATED = "DEDUPLICATED"
    SEMANTIC_FILTERED = "SEMANTIC_FILTERED"
    REVIEWED = "REVIEWED"
    FINALIZED = "FINALIZED"
    CAPTIONED = "CAPTIONED"
    VALIDATED = "VALIDATED"
