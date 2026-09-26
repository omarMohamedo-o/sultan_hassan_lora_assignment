"""Unit tests for dataset train/val/test splitting."""

from pathlib import Path

from sultan_hassan.config.models import AppConfig
from sultan_hassan.dataset.splitter import DatasetSplitter
from sultan_hassan.domain.enums import CategoryEnum, ReviewDecisionEnum
from sultan_hassan.domain.models import FinalImageRecord


def test_dataset_splitter_partitioning(test_config: AppConfig, sample_image: Path) -> None:
    splitter = DatasetSplitter(test_config)

    records = [
        FinalImageRecord(
            image_id=f"img_{i:03d}",
            source_video="sample.mp4",
            source_sha256="abc",
            frame_number=i,
            timestamp_seconds=float(i),
            original_width=1024,
            original_height=1024,
            processed_width=1024,
            processed_height=1024,
            file_path=str(sample_image),
            sha256=f"sha_{i}",
            phash=f"phash_{i}",
            category=CategoryEnum.EXTERIOR,
            review_decision=ReviewDecisionEnum.KEEP,
        )
        for i in range(20)
    ]

    partitions = splitter.partition_records(records)
    assert len(partitions["train"]) > 0
    assert len(partitions["val"]) > 0
    assert len(partitions["test"]) > 0
    assert len(partitions["train"]) + len(partitions["val"]) + len(partitions["test"]) == 20

    # Materialize splits
    summary = splitter.materialize_splits(partitions)
    assert summary["train"] == len(partitions["train"])
    assert (Path(test_config.paths.dataset_splits) / "train").exists()
