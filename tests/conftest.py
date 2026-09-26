"""Pytest fixtures for unit and integration testing."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from sultan_hassan.config.models import AppConfig, PathsConfig


@pytest.fixture
def test_dirs(tmp_path: Path) -> PathsConfig:
    """Create isolated temporary directory hierarchy."""
    paths = PathsConfig(
        raw_videos=tmp_path / "raw_videos",
        raw_external=tmp_path / "raw_external",
        frames_raw=tmp_path / "frames_raw",
        frames_quality_filtered=tmp_path / "frames_quality_filtered",
        frames_deduplicated=tmp_path / "frames_deduplicated",
        candidates_images=tmp_path / "candidates_images",
        candidates_metadata=tmp_path / "candidates_metadata",
        review_contact_sheets=tmp_path / "review_contact_sheets",
        review_decisions=tmp_path / "review_decisions",
        final_images=tmp_path / "final_images",
        final_captions=tmp_path / "final_captions",
        dataset_splits=tmp_path / "splits",
        metadata_dir=tmp_path / "metadata",
        models_lora=tmp_path / "models_lora",
        outputs_tests=tmp_path / "outputs_tests",
        outputs_evaluation=tmp_path / "outputs_evaluation",
        reports_contact_sheets=tmp_path / "reports_contact_sheets",
        reports_dataset=tmp_path / "reports_dataset",
        reports_evaluation=tmp_path / "reports_evaluation",
        deliverable=tmp_path / "deliverable",
    )
    for p in [
        paths.raw_videos,
        paths.frames_raw,
        paths.frames_quality_filtered,
        paths.frames_deduplicated,
        paths.candidates_images,
        paths.candidates_metadata,
        paths.review_contact_sheets,
        paths.review_decisions,
        paths.final_images,
        paths.final_captions,
        paths.dataset_splits,
        paths.metadata_dir,
    ]:
        p.mkdir(parents=True, exist_ok=True)
    return paths


@pytest.fixture
def test_config(test_dirs: PathsConfig) -> AppConfig:
    """Return an AppConfig targeting isolated test directories."""
    return AppConfig(paths=test_dirs)


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Create a high-resolution 1024x1024 synthetic image with sharp details."""
    img_path = tmp_path / "test_sharp_1024.jpg"
    arr = np.zeros((1024, 1024, 3), dtype=np.uint8)
    # Add contrast patterns
    arr[200:800, 200:800] = 180
    cv2.rectangle(arr, (300, 300), (700, 700), (255, 255, 255), 4)
    cv2.putText(arr, "Mamluk Architecture", (350, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    Image.fromarray(arr).save(img_path, "JPEG")
    return img_path


@pytest.fixture
def sample_small_image(tmp_path: Path) -> Path:
    """Create a sub-1024px image."""
    img_path = tmp_path / "test_small_300.jpg"
    img = Image.new("RGB", (300, 300), color=(100, 100, 100))
    img.save(img_path, "JPEG")
    return img_path


@pytest.fixture
def sample_blurry_image(tmp_path: Path) -> Path:
    """Create a blurry image failing sharpness criteria."""
    img_path = tmp_path / "test_blurry_1024.jpg"
    arr = np.ones((1024, 1024, 3), dtype=np.uint8) * 128
    blurred = cv2.GaussianBlur(arr, (51, 51), 0)
    Image.fromarray(blurred).save(img_path, "JPEG")
    return img_path


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Create a tiny 15-frame 1024x1024 synthetic test video."""
    video_path = tmp_path / "test_sultan_hassan_sample.mp4"
    fourcc = cv2.VideoWriter.fourcc("m", "p", "4", "v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (1024, 1024))
    for i in range(15):
        frame = np.zeros((1024, 1024, 3), dtype=np.uint8)
        frame[:] = (50 + i * 5, 80 + i * 5, 120)
        cv2.putText(
            frame, f"Frame {i}", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3
        )
        out.write(frame)
    out.release()
    return video_path
