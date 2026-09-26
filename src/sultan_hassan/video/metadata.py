"""Video metadata extraction and validation."""

import logging
from pathlib import Path

import cv2

from sultan_hassan.domain.exceptions import VideoInspectionError
from sultan_hassan.domain.models import VideoMetadata
from sultan_hassan.provenance.hashes import compute_file_sha256

logger = logging.getLogger(__name__)


def decode_fourcc(fourcc_val: float) -> str:
    """Decode OpenCV FourCC integer/float into a 4-letter codec string."""
    try:
        val = int(fourcc_val)
        chars = [chr((val >> 8 * i) & 0xFF) for i in range(4)]
        codec = "".join(chars).strip()
        return codec if codec.isprintable() and len(codec) > 0 else "unknown"
    except Exception:
        return "unknown"


def extract_video_metadata(video_path: Path | str) -> VideoMetadata:
    """Inspect and extract detailed metadata from a video file.

    Args:
        video_path: Path to video file.

    Returns:
        Validated VideoMetadata model instance.

    Raises:
        VideoInspectionError: If the video cannot be opened or is invalid.
    """
    path = Path(video_path)
    if not path.is_file():
        raise VideoInspectionError(f"Video file does not exist: {path}")

    # Compute SHA-256 for cryptographic provenance
    sha256 = compute_file_sha256(path)
    file_size_bytes = path.stat().st_size

    # OpenCV VideoCapture with string path
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoInspectionError(f"Failed to open video with OpenCV: {path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_val = cap.get(cv2.CAP_PROP_FOURCC)
        codec = decode_fourcc(fourcc_val)
        container = path.suffix.lower().lstrip(".")

        if fps <= 0:
            logger.warning("Video %s reported invalid FPS (%f); defaulting to 30.0", path.name, fps)
            fps = 30.0

        duration_seconds = (frame_count / fps) if frame_count > 0 else 0.0

        metadata = VideoMetadata(
            filename=path.name,
            path=str(path.resolve()),
            sha256=sha256,
            file_size_bytes=file_size_bytes,
            duration_seconds=round(duration_seconds, 2),
            width=width,
            height=height,
            fps=round(fps, 2),
            frame_count=frame_count,
            codec=codec,
            container=container,
        )
        return metadata
    finally:
        cap.release()
