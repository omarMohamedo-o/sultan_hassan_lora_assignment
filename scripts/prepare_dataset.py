"""Script to select and prepare final 25-40 images."""

from sultan_hassan.config.loader import load_config
from sultan_hassan.dataset.selector import DatasetSelector
from sultan_hassan.pipeline.state import PipelineStateManager
from sultan_hassan.utils import configure_utf8_streams


def main() -> None:
    configure_utf8_streams()

    config = load_config()
    manifest = PipelineStateManager(config).manifest
    selector = DatasetSelector(config, manifest)
    records = selector.select_and_finalize()
    print(
        f"[SUCCESS] Selected and prepared {len(records)} final images in {config.paths.final_images}"
    )


if __name__ == "__main__":
    main()
