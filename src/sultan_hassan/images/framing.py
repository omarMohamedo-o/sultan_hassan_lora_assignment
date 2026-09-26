"""Image framing, smart-cropping, and aspect ratio bucketing tool for training."""

import logging
from pathlib import Path

from PIL import Image, ImageOps

from sultan_hassan.config.models import AppConfig, FramingConfig

logger = logging.getLogger(__name__)


def find_closest_aspect_bucket(
    width: int,
    height: int,
    buckets: list[list[int]],
) -> tuple[int, int]:
    """Find the bucket with the closest aspect ratio to the source image."""
    aspect = width / max(1, height)
    best_bucket = buckets[0]
    min_diff = float("inf")

    for bw, bh in buckets:
        b_aspect = bw / bh
        diff = abs(aspect - b_aspect)
        if diff < min_diff:
            min_diff = diff
            best_bucket = [bw, bh]

    return best_bucket[0], best_bucket[1]


def frame_image(
    input_path: Path | str,
    output_path: Path | str,
    framing_config: FramingConfig | None = None,
) -> tuple[int, int]:
    """Frame, crop, pad, or bucket an image for model training readiness.

    Args:
        input_path: Source image path.
        output_path: Destination path for framed image.
        framing_config: Framing configuration (mode, buckets, target sizes).

    Returns:
        Tuple of final (width, height).
    """
    cfg = framing_config or FramingConfig()
    in_p = Path(input_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(in_p) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")

        orig_w, orig_h = img.size

        if cfg.mode == "bucket":
            # Bucket mode: pick closest standard aspect bucket
            target_w, target_h = find_closest_aspect_bucket(
                orig_w, orig_h, cfg.aspect_ratio_buckets
            )
            # Scale and crop to fit bucket exactly
            framed = ImageOps.fit(
                img,
                (target_w, target_h),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )

        elif cfg.mode == "smart_center_crop":
            # Scale and center crop to exact target resolution
            target_w, target_h = cfg.target_width, cfg.target_height
            framed = ImageOps.fit(
                img,
                (target_w, target_h),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )

        elif cfg.mode == "letterbox":
            # Pad with pad_color without cropping any content
            target_w, target_h = cfg.target_width, cfg.target_height
            scale = min(target_w / orig_w, target_h / orig_h)
            new_w = max(1, int(round(orig_w * scale)))
            new_h = max(1, int(round(orig_h * scale)))

            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            pad_color = tuple(cfg.pad_color) if len(cfg.pad_color) == 3 else (0, 0, 0)
            framed = Image.new("RGB", (target_w, target_h), color=pad_color)

            offset_x = (target_w - new_w) // 2
            offset_y = (target_h - new_h) // 2
            framed.paste(resized, (offset_x, offset_y))

        else:
            # Default fallback: scale short side to min_short_side preserving ratio
            short_side = min(orig_w, orig_h)
            if short_side < cfg.min_short_side:
                scale = cfg.min_short_side / short_side
                target_w = int(round(orig_w * scale))
                target_h = int(round(orig_h * scale))
                framed = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            else:
                framed = img
                target_w, target_h = orig_w, orig_h

        framed.save(out_p, "JPEG", quality=96, subsampling=0)
        logger.info(
            "Framed image %s -> %s (%dx%d, mode: %s)",
            in_p.name,
            out_p.name,
            target_w,
            target_h,
            cfg.mode,
        )
        return target_w, target_h


def batch_frame_images(
    source_dir: Path | str,
    output_dir: Path | str,
    config: AppConfig,
) -> list[Path]:
    """Frame all images in source_dir into output_dir according to config."""
    s_dir = Path(source_dir)
    o_dir = Path(output_dir)
    o_dir.mkdir(parents=True, exist_ok=True)

    framed_paths: list[Path] = []
    images = sorted(list(s_dir.glob("*.jpg")) + list(s_dir.glob("*.png")))

    for img_p in images:
        dest_p = o_dir / f"framed_{img_p.stem}.jpg"
        frame_image(img_p, dest_p, config.image.framing)
        framed_paths.append(dest_p)

    return framed_paths
