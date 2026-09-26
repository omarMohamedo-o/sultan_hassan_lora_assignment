"""Unit tests for configuration loading and Pydantic validation."""

from pathlib import Path

import pytest

from sultan_hassan.config.loader import deep_merge, load_config
from sultan_hassan.config.models import AppConfig
from sultan_hassan.domain.exceptions import ConfigurationError


def test_default_app_config() -> None:
    cfg = AppConfig()
    assert cfg.project.name == "sultan-hassan-flux"
    assert cfg.image.min_short_side == 1024
    assert cfg.dataset.trigger_word == "sltnhsn"
    assert cfg.dataset.split.train_ratio == 0.8


def test_deep_merge() -> None:
    source = {"video": {"sampling_mode": "fps", "fps_rate": 5.0}}
    dest = {"video": {"sampling_mode": "seconds", "sample_interval_seconds": 0.5}}
    merged = deep_merge(source, dest)
    assert merged["video"]["sampling_mode"] == "fps"
    assert merged["video"]["fps_rate"] == 5.0
    assert merged["video"]["sample_interval_seconds"] == 0.5


def test_load_real_config() -> None:
    cfg = load_config("config/config.yaml")
    assert cfg.project.name == "sultan-hassan-flux"
    assert cfg.quality.blur_threshold > 0


def test_load_missing_config_raises_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        load_config(tmp_path / "non_existent.yaml")
