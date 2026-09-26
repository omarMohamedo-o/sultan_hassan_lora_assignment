"""Unit tests for video metadata extraction and inspector."""

import shutil
from pathlib import Path

from sultan_hassan.config.models import AppConfig
from sultan_hassan.video.inspector import VideoInspector
from sultan_hassan.video.metadata import decode_fourcc, extract_video_metadata


def test_decode_fourcc() -> None:
    assert decode_fourcc(0) == "unknown"


def test_extract_video_metadata(sample_video: Path) -> None:
    meta = extract_video_metadata(sample_video)
    assert meta.filename == sample_video.name
    assert meta.width == 1024
    assert meta.height == 1024
    assert meta.frame_count == 15
    assert meta.fps == 10.0
    assert meta.duration_seconds > 0.0
    assert len(meta.sha256) == 64


def test_video_inspector_workflow(sample_video: Path, test_config: AppConfig) -> None:
    # Copy sample video into test raw videos folder
    dest = Path(test_config.paths.raw_videos) / sample_video.name
    shutil.copy(sample_video, dest)

    inspector = VideoInspector(test_config)
    videos = inspector.inspect_all()
    assert len(videos) == 1
    assert videos[0].filename == sample_video.name

    summary = inspector.get_summary(videos)
    assert summary["total_videos"] == 1
    assert summary["unique_videos"] == 1
    assert summary["total_raw_frames"] == 15
