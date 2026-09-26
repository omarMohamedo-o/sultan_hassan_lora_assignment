"""Media standardization engine ensuring all images and videos conform to pristine standards."""

import logging
from pathlib import Path

import cv2
from PIL import Image, ImageOps

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

logger = logging.getLogger(__name__)

STANDARD_IMAGE_EXT = ".jpg"
STANDARD_VIDEO_EXT = ".mp4"

SUPPORTED_IMAGE_EXTS = {".heic", ".webp", ".png", ".tiff", ".tif", ".bmp", ".jfif", ".jpeg", ".jpg"}
SUPPORTED_VIDEO_EXTS = {".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v", ".mp4"}


def verify_image_standard_quality(image_path: Path | str) -> tuple[bool, str]:
    """Verify that an image is a valid standard high-quality RGB JPEG."""
    path = Path(image_path)
    if not path.is_file():
        return False, f"File does not exist: {path}"
    if path.stat().st_size == 0:
        return False, f"Zero-byte file: {path}"
    if path.suffix.lower() != STANDARD_IMAGE_EXT:
        return False, f"Non-standard extension '{path.suffix}', expected '{STANDARD_IMAGE_EXT}'"

    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            if img.mode != "RGB":
                return False, f"Non-RGB mode '{img.mode}', expected 'RGB'"
            w, h = img.size
            if w <= 0 or h <= 0:
                return False, f"Invalid dimensions: {w}x{h}"
        return True, f"Valid standard image ({w}x{h}, {path.stat().st_size} bytes)"
    except Exception as exc:
        return False, f"Corrupted or unreadable image: {exc}"


def verify_video_standard_quality(video_path: Path | str) -> tuple[bool, str]:
    """Verify that a video is a valid standard high-quality MP4 file."""
    path = Path(video_path)
    if not path.is_file():
        return False, f"File does not exist: {path}"
    if path.stat().st_size == 0:
        return False, f"Zero-byte file: {path}"
    if path.suffix.lower() != STANDARD_VIDEO_EXT:
        return False, f"Non-standard extension '{path.suffix}', expected '{STANDARD_VIDEO_EXT}'"

    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False, f"Cannot open video stream with OpenCV: {path}"
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        if width <= 0 or height <= 0:
            return False, f"Invalid video dimensions: {width}x{height}"
        if frame_count <= 0:
            return False, f"Invalid video frame count: {frame_count}"

        return True, f"Valid standard video ({width}x{height}, {frame_count} frames, {fps:.1f} fps)"
    except Exception as exc:
        return False, f"Corrupted or unreadable video: {exc}"


def standardize_image_to_high_quality_jpg(
    input_path: Path | str,
    output_path: Path | str,
    quality: int = 96,
) -> tuple[int, int]:
    """Convert any source image (HEIC, WebP, PNG, TIFF, BMP) into standardized pristine RGB JPEG.

    - Corrects EXIF orientation.
    - Strips unsupported color profiles and alpha channels (converting RGBA/palette to pure RGB).
    - Writes with maximum visual fidelity (4:4:4 chroma subsampling=0, quality=96).
    """
    in_p = Path(input_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(in_p) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        # Force standard .jpg extension
        if out_p.suffix.lower() != STANDARD_IMAGE_EXT:
            out_p = out_p.with_suffix(STANDARD_IMAGE_EXT)

        img.save(out_p, "JPEG", quality=quality, subsampling=0)
        logger.info("Standardized image: %s -> %s (%dx%d)", in_p.name, out_p.name, w, h)
        return w, h


def standardize_video_to_high_quality_mp4(
    input_path: Path | str,
    output_path: Path | str,
) -> bool:
    """Convert video (MOV, AVI, MKV, WebM) into standardized H.264/MP4 container."""
    in_p = Path(input_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if out_p.suffix.lower() != STANDARD_VIDEO_EXT:
        out_p = out_p.with_suffix(STANDARD_VIDEO_EXT)

    cap = cv2.VideoCapture(str(in_p))
    if not cap.isOpened():
        logger.error("Failed to open video for standardization: %s", in_p.name)
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter.fourcc("m", "p", "4", "v")
    out = cv2.VideoWriter(str(out_p), fourcc, fps, (width, height))

    frames_written = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        out.write(frame)
        frames_written += 1

    cap.release()
    out.release()
    logger.info(
        "Standardized video: %s -> %s (%d frames, %dx%d, %.1ffps)",
        in_p.name,
        out_p.name,
        frames_written,
        width,
        height,
        fps,
    )
    return True


def standardize_directory_images(
    directory: Path | str,
    output_directory: Path | str | None = None,
    quality: int = 96,
    cleanup_non_standard: bool = False,
) -> list[Path]:
    """Batch convert images (.heic, .webp, .png, etc.) into standard high quality .jpg."""
    target_dir = Path(directory)
    out_dir = Path(output_directory) if output_directory else target_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files = [p for p in target_dir.iterdir() if p.is_file()]
    standardized: list[Path] = []

    for f in all_files:
        ext = f.suffix.lower()
        if ext in SUPPORTED_IMAGE_EXTS:
            dest = (out_dir / f.stem).with_suffix(STANDARD_IMAGE_EXT)
            standardize_image_to_high_quality_jpg(f, dest, quality=quality)
            valid, msg = verify_image_standard_quality(dest)
            if valid:
                standardized.append(dest)
                if cleanup_non_standard and ext != STANDARD_IMAGE_EXT:
                    try:
                        f.unlink()
                        logger.info("Cleaned up non-standard source file: %s", f.name)
                    except Exception as e:
                        logger.warning("Failed to remove source file %s: %e", f.name, e)
            else:
                logger.error("Verification failed for %s: %s", dest.name, msg)

    return standardized


def standardize_directory_videos(
    directory: Path | str,
    output_directory: Path | str | None = None,
    cleanup_non_standard: bool = False,
) -> list[Path]:
    """Batch convert non-standard videos (.mov, .avi, .mkv, .webm) into standard .mp4."""
    target_dir = Path(directory)
    out_dir = Path(output_directory) if output_directory else target_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files = [p for p in target_dir.iterdir() if p.is_file()]
    standardized: list[Path] = []

    for f in all_files:
        ext = f.suffix.lower()
        if ext in SUPPORTED_VIDEO_EXTS:
            dest = (out_dir / f.stem).with_suffix(STANDARD_VIDEO_EXT)
            if ext != STANDARD_VIDEO_EXT or out_dir != target_dir:
                success = standardize_video_to_high_quality_mp4(f, dest)
                if success:
                    valid, _ = verify_video_standard_quality(dest)
                    if valid:
                        standardized.append(dest)
                        if cleanup_non_standard and ext != STANDARD_VIDEO_EXT:
                            try:
                                f.unlink()
                                logger.info("Cleaned up non-standard video file: %s", f.name)
                            except Exception as e:
                                logger.warning("Failed to remove source video %s: %s", f.name, e)
            else:
                valid, _ = verify_video_standard_quality(f)
                if valid:
                    standardized.append(f)

    return standardized


def standardize_media_directory(
    directory: Path | str,
    output_directory: Path | str | None = None,
    quality: int = 96,
    cleanup_non_standard: bool = False,
    recursive: bool = True,
) -> dict[str, list[Path]]:
    """Recursively or directly standardize all images and videos to standard high quality formats."""
    base_dir = Path(directory)
    results: dict[str, list[Path]] = {"images": [], "videos": []}

    dirs_to_process = [base_dir]
    if recursive:
        dirs_to_process.extend([p for p in base_dir.rglob("*") if p.is_dir()])

    for d in dirs_to_process:
        out_d = Path(output_directory) / d.relative_to(base_dir) if output_directory else None
        imgs = standardize_directory_images(
            d,
            output_directory=out_d,
            quality=quality,
            cleanup_non_standard=cleanup_non_standard,
        )
        vids = standardize_directory_videos(
            d,
            output_directory=out_d,
            cleanup_non_standard=cleanup_non_standard,
        )
        results["images"].extend(imgs)
        results["videos"].extend(vids)

    return results
