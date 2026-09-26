"""Multi-signal architectural discriminator distinguishing Sultan Hassan from Al-Rifa'i Mosque."""

import logging
from pathlib import Path

import cv2
import numpy as np

from sultan_hassan.config.models import AppConfig
from sultan_hassan.domain.enums import QualityStatus, SemanticLabel
from sultan_hassan.domain.models import SemanticFilterResult
from sultan_hassan.filtering.metadata_filter import MetadataFilter
from sultan_hassan.images.quality import read_image_safely

logger = logging.getLogger(__name__)


def detect_text_and_watermark_overlay(bgr_image: np.ndarray) -> tuple[bool, float, str]:
    """Detect whether prominent text, subtitles, or watermarks occupy top/bottom bands."""
    h, w = bgr_image.shape[:2]
    # Check bottom 15% and top 12% bands where social video captions/watermarks reside
    top_band = bgr_image[0 : int(h * 0.12), :]
    bot_band = bgr_image[int(h * 0.85) : h, :]

    reasons: list[str] = []
    prominent = False
    max_density = 0.0

    for name, band in [("top_overlay", top_band), ("bottom_subtitle", bot_band)]:
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        # Text creates sharp, high gradient edge clusters
        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(np.sum(edges > 0)) / float(edges.size)
        if edge_density > 0.09:
            reasons.append(f"{name} (edge density: {edge_density:.2f})")
            prominent = True
            max_density = max(max_density, edge_density)

    return prominent, round(max_density, 3), ", ".join(reasons)


class RifaiFilter:
    """Classifies whether an image is Sultan Hassan, Al-Rifa'i, Mixed, or Uncertain."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.metadata_filter = MetadataFilter(config)
        self.clip_enabled = config.semantic.clip_enabled

    def analyze_frame(
        self,
        image_path: Path | str,
        frame_id: str,
        source_video_name: str = "",
    ) -> SemanticFilterResult:
        """Run multi-signal classification on image frame.

        Args:
            image_path: Path to candidate image.
            frame_id: Frame identifier.
            source_video_name: Name of original video for metadata context.

        Returns:
            SemanticFilterResult with architectural scores and status.
        """
        path = Path(image_path)
        img_bgr = read_image_safely(path)

        # Signal 1: Metadata score
        combined_text = f"{source_video_name} {path.stem}"
        meta_sultan, meta_rifai, meta_rationale = self.metadata_filter.score_text(combined_text)

        # Signal 2: Watermark / text detection
        has_overlay = False
        overlay_info = ""
        if img_bgr is not None:
            has_overlay, _, overlay_info = detect_text_and_watermark_overlay(img_bgr)

        # Base architectural scores derived from signals
        sultan_score = meta_sultan
        rifai_score = meta_rifai

        # Optional CLIP scoring if enabled
        if self.clip_enabled:
            # Placeholder for CLIP model inference when enabled in future GPU stages
            pass

        score_diff = round(sultan_score - rifai_score, 2)
        rationale_parts = [meta_rationale]

        # Determine semantic label and visual dominance
        if rifai_score > 0.70:
            semantic_label = SemanticLabel.AL_RIFAI
            visual_dominant = "al_rifai"
            status = QualityStatus.REJECT
            rationale_parts.append("Al-Rifa'i Mosque dominates image; strictly rejected")
        elif sultan_score > 0.70:
            semantic_label = SemanticLabel.SULTAN_HASSAN
            visual_dominant = "sultan_hassan"
            status = QualityStatus.KEEP
            rationale_parts.append("Sultan Hassan architectural features confirmed")
        elif abs(score_diff) < self.config.semantic.rifai_review_threshold:
            semantic_label = SemanticLabel.MIXED
            visual_dominant = "both"
            status = QualityStatus.REVIEW
            rationale_parts.append(
                "Ambiguous or mixed features between Sultan Hassan and Al-Rifa'i"
            )
        else:
            semantic_label = SemanticLabel.UNCERTAIN
            visual_dominant = "unknown"
            status = QualityStatus.REVIEW
            rationale_parts.append(
                "Uncertain architectural classification; queued for human review"
            )

        # Flag overlays for review or rejection if prominent
        if has_overlay:
            if status == QualityStatus.KEEP:
                status = QualityStatus.REVIEW
            rationale_parts.append(f"Detected overlay/watermark: {overlay_info}")

        return SemanticFilterResult(
            frame_id=frame_id,
            local_filename=path.name,
            sultan_hassan_score=round(sultan_score, 2),
            al_rifai_score=round(rifai_score, 2),
            score_difference=score_diff,
            semantic_label=semantic_label,
            visual_dominant=visual_dominant,
            status=status,
            rationale=" | ".join(rationale_parts),
        )
