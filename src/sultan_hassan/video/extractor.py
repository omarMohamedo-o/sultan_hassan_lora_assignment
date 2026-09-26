"""Frame extractor with configurable sampling and resumable execution."""

import logging
import re
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image

from sultan_hassan.config.models import AppConfig
from sultan_hassan.domain.enums import PipelineStage
from sultan_hassan.domain.exceptions import FrameExtractionError
from sultan_hassan.domain.models import FrameMetadata, PipelineManifest, VideoMetadata
from sultan_hassan.provenance.hashes import compute_file_sha256

logger = logging.getLogger(__name__)


def slugify_filename(name: str) -> str:
    """Create a clean filesystem-safe slug from a filename."""
    stem = Path(name).stem
    # Replace non-alphanumeric (keep arabic chars, english, numbers)
    clean = re.sub(r"[^\w\u0600-\u06FF\-]+", "_", stem).strip("_")
    return clean[:40] if clean else "video"


class FrameExtractor:
    """Extracts video frames at configurable time or FPS intervals."""

    def __init__(self, config: AppConfig, manifest: PipelineManifest | None = None) -> None:
        self.config = config
        self.output_dir = Path(config.paths.frames_raw)
        self.metadata_dir = Path(config.paths.metadata_dir)
        self.csv_path = self.metadata_dir / "frames.csv"
        self.manifest = manifest or PipelineManifest()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def extract_from_video(
        self,
        video_meta: VideoMetadata,
        skip_if_exists: bool = True,
    ) -> list[FrameMetadata]:
        """Extract frames from a single video according to sampling strategy.

        Args:
            video_meta: Validated metadata of target video.
            skip_if_exists: Skip extraction if target frame already exists and valid.

        Returns:
            List of FrameMetadata records.
        """
        cap = cv2.VideoCapture(video_meta.path)
        if not cap.isOpened():
            raise FrameExtractionError(f"Cannot open video file: {video_meta.path}")

        extracted: list[FrameMetadata] = []
        fps = video_meta.fps if video_meta.fps > 0 else 30.0
        total_frames = video_meta.frame_count
        duration = video_meta.duration_seconds

        # Determine frame indices to extract
        if self.config.video.sampling_mode == "seconds":
            interval = self.config.video.sample_interval_seconds
            timestamps = [
                i * interval
                for i in range(int(duration / interval) + 1)
                if i * interval <= duration
            ]
            target_indices = [
                min(int(t * fps), total_frames - 1) for t in timestamps if total_frames > 0
            ]
        else:
            sample_fps = self.config.video.fps_rate
            step = max(1, int(round(fps / sample_fps)))
            target_indices = list(range(0, total_frames, step))

        # Deduplicate sorted indices
        target_indices = sorted(set(target_indices))
        slug = slugify_filename(video_meta.filename)
        video_hash_short = video_meta.sha256[:8]

        logger.info(
            "Extracting up to %d frames from '%s' (duration: %.1fs)...",
            len(target_indices),
            video_meta.filename,
            duration,
        )

        for frame_idx in target_indices:
            timestamp_sec = round(frame_idx / fps, 3)
            frame_id = f"{video_hash_short}_{frame_idx:06d}"
            filename = f"{slug}_{frame_id}_{timestamp_sec:.2f}s.jpg"
            out_path = self.output_dir / filename

            # Resumability check
            if skip_if_exists and out_path.exists() and out_path.stat().st_size > 0:
                try:
                    # Validate that existing image is uncorrupted
                    with Image.open(out_path) as img:
                        w, h = img.size
                    sha256_hash = compute_file_sha256(out_path)
                    meta = FrameMetadata(
                        frame_id=frame_id,
                        video_id=video_hash_short,
                        frame_number=frame_idx,
                        timestamp_seconds=timestamp_sec,
                        width=w,
                        height=h,
                        source_video=video_meta.filename,
                        source_sha256=video_meta.sha256,
                        local_filename=filename,
                        sha256=sha256_hash,
                        stage=PipelineStage.EXTRACTED,
                    )
                    extracted.append(meta)
                    self.manifest.mark_stage(frame_id, PipelineStage.EXTRACTED, out_path)
                    continue
                except Exception:
                    logger.warning("Corrupt existing frame %s; regenerating...", filename)

            # Seek and read
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]
            # Save frame via OpenCV (use cv2.imencode for safe Unicode path writing on Windows)
            success, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if success:
                with open(out_path, "wb") as f:
                    f.write(enc.tobytes())

                sha256_hash = compute_file_sha256(out_path)
                meta = FrameMetadata(
                    frame_id=frame_id,
                    video_id=video_hash_short,
                    frame_number=frame_idx,
                    timestamp_seconds=timestamp_sec,
                    width=w,
                    height=h,
                    source_video=video_meta.filename,
                    source_sha256=video_meta.sha256,
                    local_filename=filename,
                    sha256=sha256_hash,
                    stage=PipelineStage.EXTRACTED,
                )
                extracted.append(meta)
                self.manifest.mark_stage(frame_id, PipelineStage.EXTRACTED, out_path)

        cap.release()
        return extracted

    def extract_all(
        self,
        videos: list[VideoMetadata],
        skip_duplicate_videos: bool = True,
    ) -> list[FrameMetadata]:
        """Extract frames across all provided videos and update frames.csv."""
        all_frames: list[FrameMetadata] = []
        seen_video_hashes: set[str] = set()

        for v in videos:
            if skip_duplicate_videos and v.sha256 in seen_video_hashes:
                logger.info("Skipping exact duplicate video file: %s", v.filename)
                continue
            seen_video_hashes.add(v.sha256)

            frames = self.extract_from_video(v)
            all_frames.extend(frames)

        if all_frames:
            df = pd.DataFrame([f.model_dump() for f in all_frames])
            df.to_csv(self.csv_path, index=False, encoding="utf-8")
            logger.info("Saved %d frame metadata records to %s", len(all_frames), self.csv_path)

        return all_frames
