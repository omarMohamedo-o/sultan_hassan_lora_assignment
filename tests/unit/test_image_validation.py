"""Unit tests for image validation and resolution checks."""

from pathlib import Path

from sultan_hassan.config.models import AppConfig
from sultan_hassan.images.validation import check_resolution_and_aspect, validate_image_file


def test_validate_image_file_success(sample_image: Path) -> None:
    is_valid, err, w, h = validate_image_file(sample_image)
    assert is_valid
    assert err == ""
    assert w == 1024
    assert h == 1024


def test_validate_image_file_missing(tmp_path: Path) -> None:
    is_valid, err, _, _ = validate_image_file(tmp_path / "not_found.jpg")
    assert not is_valid
    assert "File does not exist" in err


def test_check_resolution_and_aspect(test_config: AppConfig) -> None:
    # 1024x1024 should pass
    passes, _ = check_resolution_and_aspect(1024, 1024, test_config)
    assert passes

    # 300x300 should fail raw capture limit
    passes, reason = check_resolution_and_aspect(300, 300, test_config)
    assert not passes
    assert "below raw capture limit" in reason

    # 3000x600 should fail extreme aspect ratio (5.0 > 2.5) while short side >= 512
    passes, reason = check_resolution_and_aspect(3000, 600, test_config)
    assert not passes
    assert "Extreme aspect ratio" in reason
