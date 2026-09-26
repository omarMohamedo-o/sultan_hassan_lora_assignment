"""Drift monitor detecting architectural accuracy degradation and triggering retraining."""

import logging

from pydantic import BaseModel, Field

from sultan_hassan.evaluation.metrics import ArchitecturalScore, StyleBleedingResult

logger = logging.getLogger(__name__)


class DriftStatus(BaseModel):
    """Evaluation drift status and retraining recommendation."""

    accuracy_score: float = Field(ge=0.0, le=1.0)
    style_bleed_score: float = Field(ge=0.0, le=1.0)
    drift_detected: bool = False
    retrain_recommended: bool = False
    rationale: str = ""


class ModelDriftMonitor:
    """Monitors model generation metrics against baseline thresholds to trigger retraining."""

    def __init__(
        self,
        min_acceptable_accuracy: float = 0.85,
        max_acceptable_bleeding: float = 0.10,
    ) -> None:
        self.min_accuracy = min_acceptable_accuracy
        self.max_bleeding = max_acceptable_bleeding

    def evaluate_drift(
        self,
        arch_score: ArchitecturalScore,
        bleed_result: StyleBleedingResult,
    ) -> DriftStatus:
        """Check if model accuracy decreased or style bleeding increased beyond acceptable boundaries."""
        accuracy = arch_score.overall_accuracy
        bleeding = bleed_result.bleed_score

        reasons = []
        drift_detected = False

        if accuracy < self.min_accuracy:
            drift_detected = True
            reasons.append(
                f"Architectural accuracy ({accuracy:.2f}) dropped below threshold ({self.min_accuracy:.2f})"
            )

        if bleeding > self.max_bleeding:
            drift_detected = True
            reasons.append(
                f"Negative control style bleeding ({bleeding:.2f}) exceeded threshold ({self.max_bleeding:.2f})"
            )

        retrain = drift_detected

        status = DriftStatus(
            accuracy_score=round(accuracy, 3),
            style_bleed_score=round(bleeding, 3),
            drift_detected=drift_detected,
            retrain_recommended=retrain,
            rationale="; ".join(reasons) if reasons else "Model metrics within operational bounds",
        )

        if retrain:
            logger.warning(
                "[DRIFT ALERT] Performance degradation detected: %s. Retraining triggered.",
                status.rationale,
            )
        else:
            logger.info("[DRIFT OK] Model performance stable.")

        return status
