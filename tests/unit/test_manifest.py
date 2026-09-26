"""Unit tests for pipeline manifest and state tracking."""

from pathlib import Path

from sultan_hassan.domain.enums import PipelineStage
from sultan_hassan.domain.models import PipelineManifest


def test_manifest_stage_marking(tmp_path: Path) -> None:
    manifest = PipelineManifest()
    manifest.mark_stage("f_001", PipelineStage.EXTRACTED, tmp_path / "f_001.jpg")
    assert manifest.get_stage("f_001") == PipelineStage.EXTRACTED

    manifest.mark_stage("f_001", PipelineStage.QUALITY_CHECKED, tmp_path / "f_001.jpg")
    assert manifest.get_stage("f_001") == PipelineStage.QUALITY_CHECKED
