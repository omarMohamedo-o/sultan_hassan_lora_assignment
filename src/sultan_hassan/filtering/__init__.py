"""Filtering and deduplication package."""

from sultan_hassan.filtering.deduplicator import FrameDeduplicator
from sultan_hassan.filtering.metadata_filter import MetadataFilter
from sultan_hassan.filtering.rifai_filter import RifaiFilter, detect_text_and_watermark_overlay
from sultan_hassan.filtering.scoring import calculate_quality_rank_score
from sultan_hassan.filtering.semantic_filter import SemanticFilterPipeline

__all__ = [
    "FrameDeduplicator",
    "MetadataFilter",
    "RifaiFilter",
    "detect_text_and_watermark_overlay",
    "calculate_quality_rank_score",
    "SemanticFilterPipeline",
]
