"""Architectural verification criteria for Sultan Hassan Mosque."""

from sultan_hassan.evaluation.metrics import ArchitecturalScore

ARCHITECTURAL_CRITERIA = {
    "facade": "Massive plain limestone walls with high vertical recessed window bays",
    "portal": "Monumental projecting portal with towering tiered muqarnas stalactite vault",
    "courtyard": "Cruciform 4-iwan open courtyard with central octagonal wooden domed fountain",
    "minarets": "Soaring octagonal Mamluk stone minarets with carved balconies",
    "dome": "Heavy masonry dome on stepped drum over the mausoleum chamber",
}


def audit_architectural_features(features_present: list[str]) -> ArchitecturalScore:
    """Compute score based on confirmed architectural features."""
    total = len(ARCHITECTURAL_CRITERIA)
    matched = sum(
        1 for k in ARCHITECTURAL_CRITERIA if any(k in f.lower() for f in features_present)
    )
    ratio = matched / max(1, total)

    return ArchitecturalScore(
        mamluk_stonework=ratio,
        vertical_window_bays=ratio,
        monumental_portal=ratio,
        four_iwans=ratio,
        proportions_score=ratio,
        overall_accuracy=ratio,
    )
