"""Non-destructive image preprocessing for final training readiness."""

import logging
from pathlib import Path

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


def preprocess_final_image(
    input_path: Path | str,
    output_path: Path | str,
    min_short_side: int = 1024,
) -> tuple[int, int]:
    """Correct orientation, preserve geometry, convert to RGB, and save high-fidelity output.

    Args:
        input_path: Path to original source frame.
        output_path: Path where processed final image will be saved.
        min_short_side: Minimum required dimension for shorter side.

    Returns:
        Tuple of (processed_width, processed_height).
    """
    in_p = Path(input_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(in_p) as img:
        # Correct orientation based on EXIF tags
        img = ImageOps.exif_transpose(img)

        # Convert palette/RGBA images to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        orig_w, orig_h = img.size
        short_side = min(orig_w, orig_h)

        # Scale if short side is smaller than target
        if short_side < min_short_side:
            scale = min_short_side / short_side
            new_w = int(round(orig_w * scale))
            new_h = int(round(orig_h * scale))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            new_w, new_h = orig_w, orig_h

        # Save with maximum JPEG quality and sub-sampling disabled
        img.save(out_p, "JPEG", quality=96, subsampling=0)
        logger.debug("Preprocessed %s -> %s (%dx%d)", in_p.name, out_p.name, new_w, new_h)
        return new_w, new_h
