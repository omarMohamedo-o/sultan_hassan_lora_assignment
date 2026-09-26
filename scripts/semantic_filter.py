"""Script to separate Sultan Hassan frames from Al-Rifa'i Mosque frames."""

from pathlib import Path

from sultan_hassan.config.loader import load_config
from sultan_hassan.filtering.semantic_filter import SemanticFilterPipeline
from sultan_hassan.pipeline.state import PipelineStateManager
from sultan_hassan.utils import configure_utf8_streams


def main() -> None:
    configure_utf8_streams()

    config = load_config()
    manifest = PipelineStateManager(config).manifest
    dedup_dir = Path(config.paths.frames_deduplicated)
    frames = sorted(dedup_dir.glob("*.jpg"))

    pipeline = SemanticFilterPipeline(config, manifest)
    results = pipeline.filter_frames(frames)
    print(f"[SUCCESS] Classified {len(results)} frames for Sultan Hassan vs. Al-Rifa'i exclusion.")


if __name__ == "__main__":
    main()
