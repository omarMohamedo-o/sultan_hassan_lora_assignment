"""Deduplication module combining SHA-256 exact matching and perceptual hashing."""

import logging
from pathlib import Path

import pandas as pd
from PIL import Image

from sultan_hassan.config.models import AppConfig
from sultan_hassan.domain.enums import PipelineStage
from sultan_hassan.domain.models import DuplicateCluster, ImageQualityResult, PipelineManifest
from sultan_hassan.filtering.scoring import calculate_quality_rank_score
from sultan_hassan.provenance.hashes import (
    calculate_hamming_distance,
    compute_file_sha256,
    compute_image_dhash,
    compute_image_phash,
)

logger = logging.getLogger(__name__)


class FrameDeduplicator:
    """Detects exact and perceptual duplicate frames, clustering them and keeping the highest-quality frame."""

    def __init__(self, config: AppConfig, manifest: PipelineManifest | None = None) -> None:
        self.config = config
        self.output_dir = Path(config.paths.frames_deduplicated)
        self.metadata_dir = Path(config.paths.metadata_dir)
        self.csv_path = self.metadata_dir / "duplicates.csv"
        self.manifest = manifest or PipelineManifest()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def deduplicate(
        self,
        frame_paths: list[Path],
        quality_map: dict[str, ImageQualityResult] | None = None,
    ) -> tuple[list[Path], list[DuplicateCluster]]:
        """Group near-duplicate frames into clusters and pick optimal representatives.

        Args:
            frame_paths: List of candidate image paths.
            quality_map: Mapping of frame_id or filename to ImageQualityResult.

        Returns:
            Tuple of (representative_paths, clusters).
        """
        if not frame_paths:
            return [], []

        method = self.config.deduplication.method.lower()
        hash_size = self.config.deduplication.hash_size
        threshold = self.config.deduplication.distance_threshold

        self.cache_file = self.metadata_dir / "hash_cache.json"
        hash_cache: dict[str, dict[str, str]] = {}
        if self.cache_file.exists():
            try:
                import json

                hash_cache = json.loads(self.cache_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        logger.info(
            "Computing hashes for %d candidate frames (cached: %d)...",
            len(frame_paths),
            len(hash_cache),
        )
        items: list[dict[str, object]] = []
        new_cached = 0

        for p in frame_paths:
            try:
                cached = hash_cache.get(p.name)
                if cached and "sha256" in cached and "phash" in cached:
                    sha256 = cached["sha256"]
                    phash = cached["phash"]
                else:
                    sha256 = compute_file_sha256(p)
                    with Image.open(p) as img:
                        if method == "dhash":
                            phash = compute_image_dhash(img, hash_size=hash_size)
                        else:
                            phash = compute_image_phash(img, hash_size=hash_size)
                    hash_cache[p.name] = {"sha256": sha256, "phash": phash}
                    new_cached += 1

                stem_parts = p.stem.split("_")
                frame_id = "_".join(stem_parts[-2:]) if len(stem_parts) >= 2 else p.stem

                q_res = (
                    quality_map.get(frame_id) or quality_map.get(p.name) if quality_map else None
                )
                score = calculate_quality_rank_score(q_res) if q_res else 50.0

                items.append(
                    {
                        "path": p,
                        "frame_id": frame_id,
                        "sha256": sha256,
                        "phash": phash,
                        "score": score,
                    }
                )

                if new_cached > 0 and new_cached % 50 == 0:
                    import json

                    self.cache_file.write_text(json.dumps(hash_cache), encoding="utf-8")
            except Exception as exc:
                logger.error("Failed to hash frame %s: %s", p.name, exc)

        if new_cached > 0:
            import json

            self.cache_file.write_text(json.dumps(hash_cache), encoding="utf-8")

        # Step 1: Exact SHA256 clustering
        sha_groups: dict[str, list[dict[str, object]]] = {}
        for it in items:
            sha = str(it["sha256"])
            sha_groups.setdefault(sha, []).append(it)

        unique_by_sha: list[dict[str, object]] = []
        for _sha, group in sha_groups.items():
            best = max(group, key=lambda x: float(str(x["score"])))
            unique_by_sha.append(best)

        logger.info(
            "Exact SHA-256 deduplication reduced %d frames to %d unique frames",
            len(items),
            len(unique_by_sha),
        )

        # Step 2: Perceptual Hash clustering (Union-Find / greedy clustering)
        clusters: list[list[dict[str, object]]] = []
        for it in unique_by_sha:
            assigned = False
            it_phash = str(it["phash"])
            for cluster in clusters:
                # Compare against cluster representative
                rep_phash = str(cluster[0]["phash"])
                dist = calculate_hamming_distance(it_phash, rep_phash)
                if dist <= threshold:
                    cluster.append(it)
                    assigned = True
                    break
            if not assigned:
                clusters.append([it])

        # Step 3: Select representative from each cluster
        representatives: list[Path] = []
        cluster_records: list[DuplicateCluster] = []

        for idx, cluster in enumerate(clusters):
            # Select highest scoring frame in cluster
            best_frame = max(cluster, key=lambda x: float(str(x["score"])))
            best_path = Path(str(best_frame["path"]))
            best_id = str(best_frame["frame_id"])
            all_ids = [str(x["frame_id"]) for x in cluster]

            # Copy to deduplicated folder
            dest = self.output_dir / best_path.name
            if not dest.exists():
                dest.write_bytes(best_path.read_bytes())

            representatives.append(dest)
            self.manifest.mark_stage(best_id, PipelineStage.DEDUPLICATED, dest)

            cluster_model = DuplicateCluster(
                cluster_id=f"cluster_{idx:04d}",
                representative_frame_id=best_id,
                frame_ids=all_ids,
                hash_value=str(best_frame["phash"]),
                size=len(cluster),
            )
            cluster_records.append(cluster_model)

        logger.info(
            "Perceptual clustering reduced %d frames to %d deduplicated representatives (threshold=%d)",
            len(unique_by_sha),
            len(representatives),
            threshold,
        )

        # Save metadata to CSV
        if cluster_records:
            records = []
            for c in cluster_records:
                for fid in c.frame_ids:
                    records.append(
                        {
                            "cluster_id": c.cluster_id,
                            "frame_id": fid,
                            "is_representative": (fid == c.representative_frame_id),
                            "cluster_size": c.size,
                            "phash": c.hash_value,
                        }
                    )
            df = pd.DataFrame(records)
            df.to_csv(self.csv_path, index=False, encoding="utf-8")
            logger.info("Saved deduplication records to %s", self.csv_path)

        return representatives, cluster_records
