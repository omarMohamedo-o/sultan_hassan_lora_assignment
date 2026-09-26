"""Script to cluster near-duplicate frames and select optimal representatives."""

from pathlib import Path

from sultan_hassan.config.loader import load_config
from sultan_hassan.filtering.deduplicator import FrameDeduplicator
from sultan_hassan.pipeline.state import PipelineStateManager
from sultan_hassan.utils import configure_utf8_streams


def main() -> None:
    configure_utf8_streams()

    config = load_config()
    manifest = PipelineStateManager(config).manifest
    passing_dir = Path(config.paths.frames_quality_filtered)
    frame_paths = sorted(passing_dir.glob("*.jpg"))

    if not frame_paths:
        frame_paths = sorted(Path(config.paths.frames_raw).glob("*.jpg"))

    deduplicator = FrameDeduplicator(config, manifest)
    reps, clusters = deduplicator.deduplicate(frame_paths)
    print(
        f"[SUCCESS] Clustered {len(frame_paths)} frames into {len(reps)} distinct representatives."
    )


if __name__ == "__main__":
    main()
