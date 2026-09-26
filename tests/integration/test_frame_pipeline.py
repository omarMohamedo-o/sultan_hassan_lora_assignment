"""Integration test for quality filtering, deduplication, and semantic filtering."""

import shutil
from pathlib import Path

from sultan_hassan.config.models             import AppConfig
from sultan_hassan.filtering.deduplicator    import FrameDeduplicator
from sultan_hassan.filtering.semantic_filter import SemanticFilterPipeline
from sultan_hassan.images.quality            import QualityFilter


def test_frame_processing_pipeline(
    sample_image: Path, sample_small_image: Path, test_config: AppConfig
) -> None:
    # 1. Place frames in raw frames folder
    f1 = Path(test_config.paths.frames_raw) / "sample_video_frame_000001_0.50s.jpg"
    f2 = Path(test_config.paths.frames_raw) / "sample_video_frame_000002_1.00s.jpg"
    f_small = Path(test_config.paths.frames_raw) / "sample_video_frame_000003_1.50s.jpg"

    shutil.copy(sample_image, f1)
    shutil.copy(sample_image, f2)
    shutil.copy(sample_small_image, f_small)

    # 2. Quality filtering
    q_filter = QualityFilter(test_config)
    q_results = q_filter.filter_all()
    assert len(q_results) == 3

    # Small image must be rejected
    rejected = [r for r in q_results if r.frame_id == "000003_1.50s"]
    assert len(rejected) == 1
    assert rejected[0].quality_status.value == "REJECT"

    # 3. Deduplication
    passing = [
        Path(test_config.paths.frames_raw) / r.local_filename
        for r in q_results
        if r.quality_status.value != "REJECT"
    ]
    deduplicator = FrameDeduplicator(test_config)
    reps, clusters = deduplicator.deduplicate(passing)
    # Identical frames f1 and f2 must be deduplicated to 1 representative
    assert len(reps) == 1

    # 4. Semantic filter
    sem_pipe = SemanticFilterPipeline(test_config)
    sem_results = sem_pipe.filter_frames(reps)
    assert len(sem_results) == 1
    assert (Path(test_config.paths.candidates_images) / reps[0].name).exists()
