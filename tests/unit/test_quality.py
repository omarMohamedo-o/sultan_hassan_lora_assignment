"""Unit tests for blur detection and exposure quality metrics."""

from pathlib import Path

from sultan_hassan.config.models import AppConfig
from sultan_hassan.domain.enums import QualityStatus
from sultan_hassan.images.quality import evaluate_frame_quality


def test_evaluate_sharp_image(sample_image: Path, test_config: AppConfig) -> None:
    res = evaluate_frame_quality(sample_image, "frame_001", test_config)
    assert res.quality_status in (QualityStatus.KEEP, QualityStatus.REVIEW)
    assert res.width == 1024
    assert res.short_side == 1024
    assert res.blur_score > 0.0


def test_evaluate_small_image_rejected(sample_small_image: Path, test_config: AppConfig) -> None:
    res = evaluate_frame_quality(sample_small_image, "frame_small", test_config)
    assert res.quality_status == QualityStatus.REJECT
    assert "below raw capture limit" in res.rejection_reason


def test_evaluate_blurry_image(sample_blurry_image: Path, test_config: AppConfig) -> None:
    res = evaluate_frame_quality(sample_blurry_image, "frame_blur", test_config)
    assert res.blur_score < test_config.quality.blur_threshold
    assert res.quality_status in (QualityStatus.REVIEW, QualityStatus.REJECT)
