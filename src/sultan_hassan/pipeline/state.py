"""Persistent pipeline state manager."""

import json
import logging
from pathlib import Path

from sultan_hassan.config.models import AppConfig
from sultan_hassan.domain.models import PipelineManifest

logger = logging.getLogger(__name__)


class PipelineStateManager:
    """Manages loading and persisting pipeline state manifest."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.state_file = Path(config.paths.metadata_dir) / "manifest.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.manifest = self._load()

    def _load(self) -> PipelineManifest:
        """Load manifest from disk or initialize fresh manifest."""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                return PipelineManifest.model_validate(data)
            except Exception as exc:
                logger.error(
                    "Failed to parse manifest %s: %s; initializing fresh", self.state_file, exc
                )
        return PipelineManifest(project_name=self.config.project.name)

    def save(self) -> None:
        """Write current manifest state to disk."""
        try:
            data = self.manifest.model_dump()
            self.state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to write manifest %s: %s", self.state_file, exc)
