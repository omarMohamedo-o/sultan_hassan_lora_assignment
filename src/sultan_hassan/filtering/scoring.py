"""Scoring heuristics for image ranking and quality comparison."""

from sultan_hassan.domain.models import ImageQualityResult


def calculate_quality_rank_score(quality: ImageQualityResult) -> float:
    """Calculate a composite quality score for ranking cluster representatives.

    Weighting prioritizes sharpness, balanced exposure, and short-side resolution.
    """
    # Normalize blur score logarithmically
    blur_component = min(100.0, quality.blur_score)

    # Penalize extreme exposure
    brightness_penalty = 0.0
    if quality.brightness < 40.0:
        brightness_penalty = (40.0 - quality.brightness) * 1.5
    elif quality.brightness > 220.0:
        brightness_penalty = (quality.brightness - 220.0) * 1.5

    # Resolution boost
    resolution_component = min(20.0, (quality.short_side / 1024.0) * 10.0)

    score = blur_component - brightness_penalty + resolution_component
    return max(0.0, score)
