"""Semantic filter pipeline orchestrating architectural classification."""

import logging
from pathlib import Path

import pandas as pd

from sultan_hassan.config.models import AppConfig
from sultan_hassan.domain.enums import PipelineStage, QualityStatus
from sultan_hassan.domain.models import PipelineManifest, SemanticFilterResult
from sultan_hassan.filtering.rifai_filter import RifaiFilter

logger = logging.getLogger(__name__)


class SemanticFilterPipeline:
    """Evaluates frames to eliminate Al-Rifa'i Mosque, flag mixed scenes, and isolate Sultan Hassan candidates."""

    def __init__(self, config: AppConfig, manifest: PipelineManifest | None = None) -> None:
        self.config = config
        self.rifai_filter = RifaiFilter(config)
        self.candidates_dir = Path(config.paths.candidates_images)
        self.metadata_dir = Path(config.paths.metadata_dir)
        self.csv_path = self.metadata_dir / "semantic_filter.csv"
        self.manifest = manifest or PipelineManifest()
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def filter_frames(
        self,
        frame_paths: list[Path],
        source_video_map: dict[str, str] | None = None,
    ) -> list[SemanticFilterResult]:
        """Classify each frame and populate candidates directory with qualifying items."""
        results: list[SemanticFilterResult] = []

        for p in frame_paths:
            stem_parts = p.stem.split("_")
            frame_id = "_".join(stem_parts[-2:]) if len(stem_parts) >= 2 else p.stem
            source_video = (source_video_map or {}).get(frame_id, p.name)

            res = self.rifai_filter.analyze_frame(p, frame_id, source_video)
            results.append(res)

            # Pass KEEP and REVIEW to candidates for human review
            if res.status in (QualityStatus.KEEP, QualityStatus.REVIEW):
                dest = self.candidates_dir / p.name
                if not dest.exists():
                    dest.write_bytes(p.read_bytes())
                self.manifest.mark_stage(frame_id, PipelineStage.SEMANTIC_FILTERED, dest)

        if results:
            df = pd.DataFrame([r.model_dump() for r in results])
            df.to_csv(self.csv_path, index=False, encoding="utf-8")
            logger.info("Saved %d semantic filter records to %s", len(results), self.csv_path)

        return results
