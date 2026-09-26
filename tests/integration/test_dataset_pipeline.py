"""Integration test for final dataset selection, captions, validation, and splitting."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from sultan_hassan.config.models import AppConfig
from sultan_hassan.dataset.selector import DatasetSelector
from sultan_hassan.dataset.splitter import DatasetSplitter
from sultan_hassan.dataset.validator import DatasetValidator
from sultan_hassan.domain.enums import CategoryEnum, ReviewDecisionEnum
from sultan_hassan.review.decisions import DecisionStore


def test_final_dataset_pipeline(sample_image: Path, test_config: AppConfig) -> None:
    # Relax dataset minimum count for fast integration test
    test_config.dataset.min_images = 3
    test_config.dataset.max_images = 10

    # 1. Place 3 candidate images
    cand_dir = Path(test_config.paths.candidates_images)
    store = DecisionStore(test_config)

    for i in range(1, 4):
        cname = f"candidate_frame_{i:06d}.jpg"
        cdest = cand_dir / cname
        arr = np.zeros((1024, 1024, 3), dtype=np.uint8)
        arr[100:900, 100:900] = 100 + i * 40
        cv2.putText(
            arr, f"Facade {i}", (200, 500), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3
        )
        Image.fromarray(arr).save(cdest, "JPEG")
        fid = f"frame_{i:06d}"
        store.save_decision(
            frame_id=fid,
            decision=ReviewDecisionEnum.KEEP,
            category=CategoryEnum.EXTERIOR,
        )

    # 2. Select and finalize
    selector = DatasetSelector(test_config)
    final_recs = selector.select_and_finalize()
    assert len(final_recs) == 3

    # Verify images and captions exist
    final_imgs = list(Path(test_config.paths.final_images).glob("*.jpg"))
    final_caps = list(Path(test_config.paths.final_captions).glob("*.txt"))
    assert len(final_imgs) == 3
    assert len(final_caps) == 3

    # 3. Validate dataset
    validator = DatasetValidator(test_config)
    val_res = validator.validate(raise_on_failure=False)
    assert val_res.is_valid
    assert val_res.total_images == 3

    # 4. Split dataset into train/val/test
    splitter = DatasetSplitter(test_config)
    partitions = splitter.partition_records(final_recs)
    split_summary = splitter.materialize_splits(partitions)
    assert "train" in split_summary
    assert (Path(test_config.paths.dataset_splits) / "train").exists()
