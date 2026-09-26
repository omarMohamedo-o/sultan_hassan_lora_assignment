"""Video inspector module for discovering and auditing raw video assets."""

import logging
from pathlib import Path

import pandas as pd

from sultan_hassan.config.models import AppConfig
from sultan_hassan.domain.models import VideoMetadata
from sultan_hassan.video.metadata import extract_video_metadata

logger = logging.getLogger(__name__)


class VideoInspector:
    """Discovers, validates, and records metadata for raw video files."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.raw_dir = Path(config.paths.raw_videos)
        self.metadata_dir = Path(config.paths.metadata_dir)
        self.csv_path = self.metadata_dir / "videos.csv"
        self.supported_exts = set(config.video.supported_extensions)

    def discover_videos(self) -> list[Path]:
        """Find all supported video files in the raw video directory."""
        if not self.raw_dir.exists():
            return []
        videos = [
            p
            for p in self.raw_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.supported_exts
        ]
        return sorted(videos, key=lambda x: x.name)

    def inspect_all(self) -> list[VideoMetadata]:
        """Inspect all discovered videos, save metadata to CSV, and log summary.

        Returns:
            List of validated VideoMetadata models.
        """
        video_files = self.discover_videos()
        results: list[VideoMetadata] = []
        seen_hashes: dict[str, str] = {}

        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        for vpath in video_files:
            try:
                meta = extract_video_metadata(vpath)
                if meta.sha256 in seen_hashes:
                    logger.warning(
                        "Duplicate video detected: '%s' is identical to '%s' (SHA256: %s)",
                        meta.filename,
                        seen_hashes[meta.sha256],
                        meta.sha256[:12],
                    )
                else:
                    seen_hashes[meta.sha256] = meta.filename

                results.append(meta)
            except Exception as exc:
                logger.error("Failed to inspect video %s: %s", vpath.name, exc)

        if results:
            df = pd.DataFrame([m.model_dump() for m in results])
            df.to_csv(self.csv_path, index=False, encoding="utf-8")
            logger.info("Saved video metadata to %s (%d records)", self.csv_path, len(results))

        return results

    def get_summary(self, metadata_list: list[VideoMetadata]) -> dict[str, object]:
        """Compute aggregate summary metrics across inspected videos."""
        total_videos = len(metadata_list)
        unique_hashes = {m.sha256 for m in metadata_list}
        total_duration = sum(m.duration_seconds for m in metadata_list)
        total_frames = sum(m.frame_count for m in metadata_list)
        resolutions = sorted({f"{m.width}x{m.height}" for m in metadata_list})

        return {
            "total_videos": total_videos,
            "unique_videos": len(unique_hashes),
            "duplicate_videos": total_videos - len(unique_hashes),
            "total_duration_seconds": round(total_duration, 2),
            "total_raw_frames": total_frames,
            "resolutions": resolutions,
        }
