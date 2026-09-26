"""Automated image quality assessment for blur, exposure, contrast, and resolution."""

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sultan_hassan.config.models import AppConfig
from sultan_hassan.domain.enums import PipelineStage, QualityStatus
from sultan_hassan.domain.models import ImageQualityResult, PipelineManifest
from sultan_hassan.images.validation import check_resolution_and_aspect, validate_image_file

logger = logging.getLogger(__name__)


def read_image_safely(image_path: Path | str) -> np.ndarray | None:
    """Read image into a BGR numpy array safely supporting non-ASCII / Unicode paths on Windows."""
    path = Path(image_path)
    if not path.is_file():
        return None
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as exc:
        logger.debug("Failed to read image %s via imdecode: %s", path.name, exc)
        return None


def calculate_blur_score(gray: np.ndarray) -> float:
    """Calculate the variance of the Laplacian as a focus/blur metric."""
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def calculate_exposure_metrics(gray: np.ndarray) -> tuple[float, float]:
    """Calculate mean brightness and standard deviation (contrast) of pixel intensities."""
    mean_brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    return round(mean_brightness, 2), round(contrast, 2)


def evaluate_frame_quality(
    image_path: Path | str,
    frame_id: str,
    config: AppConfig,
) -> ImageQualityResult:
    """Assess a frame against resolution, blur, brightness, contrast, and aspect ratio criteria.

    Args:
        image_path: Path to the image file.
        frame_id: Unique identifier for the frame.
        config: Application configuration with threshold limits.

    Returns:
        ImageQualityResult with detailed metrics and KEEP/REVIEW/REJECT classification.
    """
    path = Path(image_path)
    is_valid, err_msg, w, h = validate_image_file(path)
    if not is_valid:
        return ImageQualityResult(
            frame_id=frame_id,
            local_filename=path.name,
            width=0,
            height=0,
            short_side=0,
            blur_score=0.0,
            brightness=0.0,
            contrast=0.0,
            aspect_ratio=0.0,
            quality_status=QualityStatus.REJECT,
            rejection_reason=f"Corrupt/Invalid file: {err_msg}",
        )

    short_side = min(w, h)
    aspect_ratio = round(max(w, h) / max(1, min(w, h)), 2)

    # Check resolution against assessment gate (1024px short side)
    passes_res, res_reason = check_resolution_and_aspect(w, h, config)
    if not passes_res:
        return ImageQualityResult(
            frame_id=frame_id,
            local_filename=path.name,
            width=w,
            height=h,
            short_side=short_side,
            blur_score=0.0,
            brightness=0.0,
            contrast=0.0,
            aspect_ratio=aspect_ratio,
            quality_status=QualityStatus.REJECT,
            rejection_reason=res_reason,
        )

    # Read image matrix
    bgr = read_image_safely(path)
    if bgr is None:
        return ImageQualityResult(
            frame_id=frame_id,
            local_filename=path.name,
            width=w,
            height=h,
            short_side=short_side,
            blur_score=0.0,
            brightness=0.0,
            contrast=0.0,
            aspect_ratio=aspect_ratio,
            quality_status=QualityStatus.REJECT,
            rejection_reason="Failed to decode image data into matrix",
        )

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = round(calculate_blur_score(gray), 2)
    brightness, contrast = calculate_exposure_metrics(gray)

    # Classify based on thresholds
    q_cfg = config.quality
    reasons: list[str] = []
    status = QualityStatus.KEEP

    # Check resolution against target (e.g. 720p vertical video)
    if short_side < config.image.min_short_side:
        status = QualityStatus.REVIEW
        reasons.append(
            f"Short side {short_side}px < {config.image.min_short_side}px (eligible for framing upscale)"
        )

    # Severe blur rejection vs borderline review
    if blur < (q_cfg.blur_threshold * 0.4):
        status = QualityStatus.REJECT
        reasons.append(f"Severe blur (score: {blur} < {q_cfg.blur_threshold * 0.4:.1f})")
    elif blur < q_cfg.blur_threshold:
        if status != QualityStatus.REJECT:
            status = QualityStatus.REVIEW
        reasons.append(f"Borderline blur (score: {blur} < {q_cfg.blur_threshold})")

    # Dark / Underexposed
    if brightness < (q_cfg.brightness_min * 0.5):
        status = QualityStatus.REJECT
        reasons.append(
            f"Extremely dark (brightness: {brightness} < {q_cfg.brightness_min * 0.5:.1f})"
        )
    elif brightness < q_cfg.brightness_min:
        if status != QualityStatus.REJECT:
            status = QualityStatus.REVIEW
        reasons.append(f"Borderline low brightness ({brightness} < {q_cfg.brightness_min})")

    # Overexposed
    if brightness > (q_cfg.brightness_max + 15):
        status = QualityStatus.REJECT
        reasons.append(f"Severe overexposure (brightness: {brightness})")
    elif brightness > q_cfg.brightness_max:
        if status != QualityStatus.REJECT:
            status = QualityStatus.REVIEW
        reasons.append(f"High brightness ({brightness} > {q_cfg.brightness_max})")

    # Very low contrast / blank frame
    if contrast < (q_cfg.contrast_min * 0.4):
        status = QualityStatus.REJECT
        reasons.append(f"Blank / flat frame (contrast: {contrast})")
    elif contrast < q_cfg.contrast_min:
        if status != QualityStatus.REJECT:
            status = QualityStatus.REVIEW
        reasons.append(f"Low contrast ({contrast} < {q_cfg.contrast_min})")

    return ImageQualityResult(
        frame_id=frame_id,
        local_filename=path.name,
        width=w,
        height=h,
        short_side=short_side,
        blur_score=blur,
        brightness=brightness,
        contrast=contrast,
        aspect_ratio=aspect_ratio,
        quality_status=status,
        rejection_reason="; ".join(reasons),
    )


class QualityFilter:
    """Batch quality filter that evaluates raw frames and records results."""

    def __init__(self, config: AppConfig, manifest: PipelineManifest | None = None) -> None:
        self.config = config
        self.raw_frames_dir = Path(config.paths.frames_raw)
        self.filtered_dir = Path(config.paths.frames_quality_filtered)
        self.metadata_dir = Path(config.paths.metadata_dir)
        self.csv_path = self.metadata_dir / "quality.csv"
        self.manifest = manifest or PipelineManifest()
        self.filtered_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def filter_all(
        self,
        frame_paths: list[Path] | None = None,
        batch_size: int = 20,
    ) -> list[ImageQualityResult]:
        """Assess quality across all raw frames, saving in batches to allow stopping and resuming anytime."""
        if frame_paths is None:
            if not self.raw_frames_dir.exists():
                return []
            frame_paths = sorted([p for p in self.raw_frames_dir.glob("*.jpg") if p.is_file()])

        # Load existing results for resumability
        existing: dict[str, ImageQualityResult] = {}
        if self.csv_path.exists():
            try:
                df_ex = pd.read_csv(self.csv_path, encoding="utf-8").fillna("")
                for _, row in df_ex.iterrows():
                    rec = ImageQualityResult.model_validate(row.to_dict())
                    existing[rec.frame_id] = rec
            except Exception as exc:
                logger.debug("Could not read previous quality records: %s", exc)

        results: list[ImageQualityResult] = []
        newly_processed = 0

        for p in frame_paths:
            stem_parts = p.stem.split("_")
            frame_id = "_".join(stem_parts[-2:]) if len(stem_parts) >= 2 else p.stem

            if frame_id in existing:
                res = existing[frame_id]
            else:
                res = evaluate_frame_quality(p, frame_id, self.config)
                existing[frame_id] = res
                newly_processed += 1

            results.append(res)

            # If KEEP or REVIEW, save reference into quality_filtered
            if res.quality_status in (QualityStatus.KEEP, QualityStatus.REVIEW):
                dest = self.filtered_dir / p.name
                if not dest.exists():
                    try:
                        dest.write_bytes(p.read_bytes())
                    except Exception as exc:
                        logger.error("Failed to copy passing frame %s: %s", p.name, exc)
                self.manifest.mark_stage(frame_id, PipelineStage.QUALITY_CHECKED, dest)

            # Checkpoint flush in batches
            if newly_processed > 0 and newly_processed % batch_size == 0:
                df = pd.DataFrame([r.model_dump() for r in existing.values()])
                df.to_csv(self.csv_path, index=False, encoding="utf-8")
                logger.debug("Flushed %d quality records to %s", len(existing), self.csv_path)

        if existing:
            df = pd.DataFrame([r.model_dump() for r in existing.values()])
            df.to_csv(self.csv_path, index=False, encoding="utf-8")
            logger.info("Saved %d quality inspection records to %s", len(existing), self.csv_path)

        return results
