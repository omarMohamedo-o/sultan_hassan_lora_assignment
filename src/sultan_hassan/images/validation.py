"""Image validation checks for corruption, resolution, and geometric constraints."""

from pathlib import Path

from PIL import Image

from sultan_hassan.config.models import AppConfig


def validate_image_file(image_path: Path | str) -> tuple[bool, str, int, int]:
    """Verify that an image file can be opened, decoded, and check dimensions.

    Args:
        image_path: Path to target image.

    Returns:
        Tuple of (is_valid, error_reason, width, height).
    """
    path = Path(image_path)
    if not path.is_file():
        return False, "File does not exist", 0, 0

    try:
        with Image.open(path) as img:
            img.verify()
        # Re-open for dimension inspection after verify()
        with Image.open(path) as img:
            w, h = img.size
            if w <= 0 or h <= 0:
                return False, "Invalid image dimensions", 0, 0
            return True, "", w, h
    except Exception as exc:
        return False, f"Corrupted image file: {exc}", 0, 0


def check_resolution_and_aspect(
    width: int,
    height: int,
    config: AppConfig,
) -> tuple[bool, str]:
    """Check whether resolution meets >= 1024px short side and aspect ratio limits.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        config: Application configuration.

    Returns:
        Tuple of (passes_threshold, failure_reason).
    """
    short_side = min(width, height)
    min_raw = config.quality.min_raw_short_side

    if short_side < min_raw:
        return False, f"Short side {short_side}px is below raw capture limit {min_raw}px"

    aspect_ratio = max(width, height) / max(1, min(width, height))
    if aspect_ratio > config.quality.max_aspect_ratio:
        return (
            False,
            f"Extreme aspect ratio ({aspect_ratio:.2f} > {config.quality.max_aspect_ratio})",
        )

    return True, ""
