"""Integration test for video ingestion and frame extraction."""

import shutil
from pathlib import Path

from sultan_hassan.config.models   import AppConfig
from sultan_hassan.video.extractor import FrameExtractor
from sultan_hassan.video.inspector import VideoInspector


def test_video_to_frames_integration(sample_video: Path, test_config: AppConfig) -> None:
    # 1. Place video in raw videos
    vdest = Path(test_config.paths.raw_videos) / sample_video.name
    shutil.copy(sample_video, vdest)

    # 2. Inspect
    inspector = VideoInspector(test_config)
    videos = inspector.inspect_all()
    assert len(videos) == 1

    # 3. Extract frames
    extractor = FrameExtractor(test_config)
    frames = extractor.extract_all(videos)
    assert len(frames) > 0

    csv_file = Path(test_config.paths.metadata_dir) / "frames.csv"
    assert csv_file.exists()
