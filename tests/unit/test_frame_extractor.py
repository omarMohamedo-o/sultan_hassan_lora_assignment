"""Unit tests for frame extractor and sampling."""

from pathlib import Path

from sultan_hassan.config.models import AppConfig
from sultan_hassan.video.extractor import FrameExtractor, slugify_filename
from sultan_hassan.video.metadata import extract_video_metadata


def test_slugify_filename() -> None:
    slug = slugify_filename("#مسجد_السلطان_حسن (1).mp4")
    assert "مسجد_السلطان_حسن" in slug


def test_frame_extraction_and_resumability(sample_video: Path, test_config: AppConfig) -> None:
    test_config.video.sampling_mode = "seconds"
    test_config.video.sample_interval_seconds = 0.5

    meta = extract_video_metadata(sample_video)
    extractor = FrameExtractor(test_config)

    # Initial extraction
    frames1 = extractor.extract_from_video(meta, skip_if_exists=True)
    assert len(frames1) > 0
    frame_files1 = list(Path(test_config.paths.frames_raw).glob("*.jpg"))
    assert len(frame_files1) == len(frames1)

    # Re-run extraction should skip existing without regenerating
    frames2 = extractor.extract_from_video(meta, skip_if_exists=True)
    assert len(frames2) == len(frames1)
