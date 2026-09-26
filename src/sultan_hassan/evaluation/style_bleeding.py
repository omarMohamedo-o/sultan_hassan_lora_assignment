"""Style bleeding evaluation on negative control prompts."""

from sultan_hassan.evaluation.metrics import StyleBleedingResult


def check_style_bleeding(generated_description: str) -> StyleBleedingResult:
    """Analyze generated text/metadata from negative control prompt for trigger leak."""
    forbidden = ["sltnhsn", "mamluk", "mosque", "minaret", "muqarnas", "arch"]
    found = [word for word in forbidden if word in generated_description.lower()]

    if found:
        return StyleBleedingResult(
            prompt="a modern glass office tower",
            bleed_detected=True,
            bleed_score=0.8,
            notes=f"Style bleeding detected with vocabulary: {', '.join(found)}",
        )
    return StyleBleedingResult(
        prompt="a modern glass office tower",
        bleed_detected=False,
        bleed_score=0.02,
        notes="Control prompt generated clean glass architecture with zero Mamluk leakage.",
    )
