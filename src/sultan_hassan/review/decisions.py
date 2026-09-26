"""Review decision store and persistence."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from sultan_hassan.config.models import AppConfig
from sultan_hassan.domain.enums import CategoryEnum, ReviewDecisionEnum
from sultan_hassan.domain.models import ReviewDecision

logger = logging.getLogger(__name__)


class DecisionStore:
    """Persistent storage for human review decisions."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.decisions_dir = Path(config.paths.review_decisions)
        self.metadata_dir = Path(config.paths.metadata_dir)
        self.json_path = self.decisions_dir / "decisions.json"
        self.csv_path = self.metadata_dir / "review.csv"
        self.decisions_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.decisions: dict[str, ReviewDecision] = self._load()

    def _load(self) -> dict[str, ReviewDecision]:
        """Load existing decisions from disk for resumable review."""
        if not self.json_path.exists():
            return {}
        try:
            data = json.loads(self.json_path.read_text(encoding="utf-8"))
            return {k: ReviewDecision.model_validate(v) for k, v in data.items()}
        except Exception as exc:
            logger.error("Failed to load existing review decisions: %s", exc)
            return {}

    def save_decision(
        self,
        frame_id: str,
        decision: ReviewDecisionEnum,
        category: CategoryEnum = CategoryEnum.OTHER,
        reviewer: str = "human",
        notes: str = "",
    ) -> ReviewDecision:
        """Record or update a review decision immediately to JSON and CSV."""
        rec = ReviewDecision(
            frame_id=frame_id,
            decision=decision,
            category=category,
            reviewer=reviewer,
            notes=notes,
            timestamp=datetime.now(UTC).isoformat(),
        )
        self.decisions[frame_id] = rec

        # Save JSON
        data = {k: v.model_dump() for k, v in self.decisions.items()}
        self.json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        # Save CSV
        records = [v.model_dump() for v in self.decisions.values()]
        df = pd.DataFrame(records)
        df.to_csv(self.csv_path, index=False, encoding="utf-8")

        logger.info(
            "Saved review decision for frame %s: %s (%s)", frame_id, decision.value, category.value
        )
        return rec

    def get_decision(self, frame_id: str) -> ReviewDecision | None:
        """Retrieve existing decision for a frame."""
        return self.decisions.get(frame_id)

    def is_reviewed(self, frame_id: str) -> bool:
        """Check if frame already has an recorded human decision."""
        return frame_id in self.decisions
