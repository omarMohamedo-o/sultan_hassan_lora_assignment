"""Unit tests for image framing, bucketing, and padding."""

from pathlib import Path

from sultan_hassan.config.models import FramingConfig
from sultan_hassan.images.framing import find_closest_aspect_bucket, frame_image


def test_find_closest_aspect_bucket() -> None:
    buckets = [[1024, 1024], [896, 1152], [1152, 896]]
    # Square image -> [1024, 1024]
    w, h = find_closest_aspect_bucket(1000, 1000, buckets)
    assert (w, h) == (1024, 1024)

    # Portrait image -> [896, 1152]
    w, h = find_closest_aspect_bucket(900, 1200, buckets)
    assert (w, h) == (896, 1152)


def test_frame_image_letterbox(sample_image: Path, tmp_path: Path) -> None:
    cfg = FramingConfig(mode="letterbox", target_width=1024, target_height=1024)
    out_file = tmp_path / "letterboxed.jpg"
    w, h = frame_image(sample_image, out_file, cfg)
    assert (w, h) == (1024, 1024)
    assert out_file.exists()


def test_frame_image_bucket(sample_image: Path, tmp_path: Path) -> None:
    cfg = FramingConfig(mode="bucket")
    out_file = tmp_path / "bucketed.jpg"
    w, h = frame_image(sample_image, out_file, cfg)
    assert (w, h) == (1024, 1024)
    assert out_file.exists()
