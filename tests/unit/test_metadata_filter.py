"""Unit tests for metadata and Al-Rifa'i architectural discrimination."""

from pathlib import Path

from sultan_hassan.config.models import AppConfig
from sultan_hassan.domain.enums import QualityStatus, SemanticLabel
from sultan_hassan.filtering.metadata_filter import MetadataFilter
from sultan_hassan.filtering.rifai_filter import RifaiFilter


def test_metadata_filter_keywords(test_config: AppConfig) -> None:
    mf = MetadataFilter(test_config)

    sultan_s, rifai_s, _ = mf.score_text("مسجد السلطان حسن في القاهرة")
    assert sultan_s > rifai_s

    sultan_s, rifai_s, _ = mf.score_text("مسجد الرفاعي بالقاهرة")
    assert rifai_s > sultan_s


def test_rifai_filter_rejection(sample_image: Path, test_config: AppConfig) -> None:
    rf = RifaiFilter(test_config)

    # Frame indicating Al-Rifa'i Mosque in title must be rejected
    res_rifai = rf.analyze_frame(sample_image, "f_rifai", source_video_name="مسجد_الرفاعي.mp4")
    assert res_rifai.semantic_label == SemanticLabel.AL_RIFAI
    assert res_rifai.status == QualityStatus.REJECT

    # Frame indicating Sultan Hassan must be kept
    res_sultan = rf.analyze_frame(
        sample_image, "f_sultan", source_video_name="مسجد_السلطان_حسن.mp4"
    )
    assert res_sultan.semantic_label == SemanticLabel.SULTAN_HASSAN
    assert res_sultan.status in (QualityStatus.KEEP, QualityStatus.REVIEW)
