"""Unit tests for dataset validator and quality gates."""

from pathlib import Path

from sultan_hassan.config.models import AppConfig
from sultan_hassan.dataset.validator import DatasetValidator


def test_validator_fails_underpopulated_dataset(test_config: AppConfig) -> None:
    validator = DatasetValidator(test_config)
    res = validator.validate(raise_on_failure=False)
    assert not res.is_valid
    assert any("below the minimum" in err for err in res.errors)


def test_validator_detects_orphan_captions(test_config: AppConfig, sample_image: Path) -> None:
    validator = DatasetValidator(test_config)
    # Add an orphan caption
    cap_file = Path(test_config.paths.final_captions) / "orphan_sample.txt"
    cap_file.write_text("sltnhsn, massive stone wall", encoding="utf-8")

    res = validator.validate(raise_on_failure=False)
    assert any("without matching images" in err for err in res.errors)
