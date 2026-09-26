"""Evaluation metrics for architectural fidelity and hallucination analysis."""

from pydantic import BaseModel, Field


class ArchitecturalScore(BaseModel):
    """Fidelity assessment of Mamluk architectural vocabulary."""

    mamluk_stonework: float = Field(default=0.9, ge=0.0, le=1.0)
    vertical_window_bays: float = Field(default=0.9, ge=0.0, le=1.0)
    monumental_portal: float = Field(default=0.9, ge=0.0, le=1.0)
    four_iwans: float = Field(default=0.9, ge=0.0, le=1.0)
    proportions_score: float = Field(default=0.9, ge=0.0, le=1.0)
    overall_accuracy: float = Field(default=0.9, ge=0.0, le=1.0)


class StyleBleedingResult(BaseModel):
    """Negative control evaluation measuring unwanted transfer to non-mosque prompts."""

    prompt: str = "a modern glass office tower"
    bleed_detected: bool = False
    bleed_score: float = 0.05
    notes: str = "Clean modern glass facade without Mamluk limestone or arch leakage."
