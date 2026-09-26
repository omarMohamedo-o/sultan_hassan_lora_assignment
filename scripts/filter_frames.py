"""Script to filter extracted frames by quality, blur, and resolution."""

from sultan_hassan.config.loader import load_config
from sultan_hassan.domain.enums import QualityStatus
from sultan_hassan.images.quality import QualityFilter
from sultan_hassan.pipeline.state import PipelineStateManager
from sultan_hassan.utils import configure_utf8_streams


def main() -> None:
    configure_utf8_streams()

    config = load_config()
    manifest = PipelineStateManager(config).manifest
    q_filter = QualityFilter(config, manifest)
    results = q_filter.filter_all()

    keep = sum(1 for r in results if r.quality_status == QualityStatus.KEEP)
    review = sum(1 for r in results if r.quality_status == QualityStatus.REVIEW)
    reject = sum(1 for r in results if r.quality_status == QualityStatus.REJECT)

    print(f"[SUCCESS] Filtered {len(results)} frames:")
    print(f"  - KEEP: {keep}")
    print(f"  - REVIEW: {review}")
    print(f"  - REJECT: {reject}")


if __name__ == "__main__":
    main()
