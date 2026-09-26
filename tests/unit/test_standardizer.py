"""Unit tests for media standardizer."""

from pathlib import Path

from PIL import Image

from sultan_hassan.images.standardizer import (
    standardize_image_to_high_quality_jpg,
    standardize_video_to_high_quality_mp4,
)


def test_standardize_png_to_high_quality_jpg(tmp_path: Path) -> None:
    # Create RGBA PNG with transparency
    png_path = tmp_path / "sample_rgba.png"
    img = Image.new("RGBA", (1024, 1024), color=(100, 150, 200, 128))
    img.save(png_path, "PNG")

    out_jpg = tmp_path / "standardized.jpg"
    w, h = standardize_image_to_high_quality_jpg(png_path, out_jpg)

    assert (w, h) == (1024, 1024)
    assert out_jpg.exists()
    with Image.open(out_jpg) as res_img:
        assert res_img.format == "JPEG"
        assert res_img.mode == "RGB"


def test_standardize_video(sample_video: Path, tmp_path: Path) -> None:
    out_mp4 = tmp_path / "standardized_video.mp4"
    success = standardize_video_to_high_quality_mp4(sample_video, out_mp4)
    assert success
    assert out_mp4.exists()


def test_verify_image_standard_quality(tmp_path: Path) -> None:
    from sultan_hassan.images.standardizer import verify_image_standard_quality

    valid_jpg = tmp_path / "valid.jpg"
    img = Image.new("RGB", (512, 512), color=(200, 100, 50))
    img.save(valid_jpg, "JPEG")

    is_valid, msg = verify_image_standard_quality(valid_jpg)
    assert is_valid
    assert "Valid standard image" in msg

    # Non-standard extension test
    png_path = tmp_path / "invalid.png"
    img.save(png_path, "PNG")
    is_valid_png, msg_png = verify_image_standard_quality(png_path)
    assert not is_valid_png
    assert "Non-standard extension" in msg_png


def test_verify_video_standard_quality(sample_video: Path) -> None:
    from sultan_hassan.images.standardizer import verify_video_standard_quality

    is_valid, msg = verify_video_standard_quality(sample_video)
    assert is_valid
    assert "Valid standard video" in msg


def test_standardize_media_directory(tmp_path: Path, sample_video: Path) -> None:
    import shutil

    from sultan_hassan.images.standardizer import standardize_media_directory

    # Create dummy directory with png and video
    media_dir = tmp_path / "raw_media"
    media_dir.mkdir()

    png_path = media_dir / "test_photo.png"
    Image.new("RGB", (300, 300), color=(10, 20, 30)).save(png_path, "PNG")

    vid_path = media_dir / "test_video.mp4"
    shutil.copy(sample_video, vid_path)

    res = standardize_media_directory(media_dir, cleanup_non_standard=True)
    assert len(res["images"]) >= 1
    assert len(res["videos"]) >= 1

    # Check that png was cleaned up and jpg exists
    assert not png_path.exists()
    assert (media_dir / "test_photo.jpg").exists()
