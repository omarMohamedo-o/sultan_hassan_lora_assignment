"""Evaluation and architectural validation package."""

from sultan_hassan.evaluation.architecture import (
    ARCHITECTURAL_CRITERIA,
    audit_architectural_features,
)
from sultan_hassan.evaluation.metrics import ArchitecturalScore, StyleBleedingResult
from sultan_hassan.evaluation.report import EvaluationReporter
from sultan_hassan.evaluation.style_bleeding import check_style_bleeding

__all__ = [
    "ArchitecturalScore",
    "StyleBleedingResult",
    "ARCHITECTURAL_CRITERIA",
    "audit_architectural_features",
    "check_style_bleeding",
    "EvaluationReporter",
]
