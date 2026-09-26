"""Script to extract frames from inspected videos."""

from sultan_hassan.config.loader import load_config
from sultan_hassan.pipeline.state import PipelineStateManager
from sultan_hassan.utils import configure_utf8_streams
from sultan_hassan.video.extractor import FrameExtractor
from sultan_hassan.video.inspector import VideoInspector


def main() -> None:
    configure_utf8_streams()

    config = load_config()
    manifest = PipelineStateManager(config).manifest
    inspector = VideoInspector(config)
    videos = inspector.inspect_all()

    extractor = FrameExtractor(config, manifest)
    frames = extractor.extract_all(videos)
    print(f"[SUCCESS] Extracted {len(frames)} frames across {len(videos)} videos.")


if __name__ == "__main__":
    main()
