"""Evaluation report generator."""

import html
from pathlib import Path

from sultan_hassan.evaluation.metrics import ArchitecturalScore, StyleBleedingResult


class EvaluationReporter:
    """Generates evaluation summaries and HTML reports."""

    def __init__(self, output_dir: Path | str = "outputs/evaluation") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_report(
        self,
        arch_score: ArchitecturalScore,
        bleed_res: StyleBleedingResult,
        filename: str = "evaluation_report.html",
    ) -> Path:
        out_p = self.output_dir / filename
        html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Model Evaluation - Sultan Hassan FLUX LoRA</title>
    <style>
        body {{ background: #0d1117; color: #c9d1d9; font-family: sans-serif; padding: 32px; }}
        h1 {{ color: #58a6ff; }}
        .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
    </style>
</head>
<body>
    <h1>Sultan Hassan LoRA Architectural Evaluation</h1>
    <div class="card">
        <h3>Architectural Fidelity: {arch_score.overall_accuracy * 100:.1f}%</h3>
        <p>Mamluk stonework: {arch_score.mamluk_stonework:.2f}</p>
        <p>Portal & Muqarnas: {arch_score.monumental_portal:.2f}</p>
        <p>Courtyard Iwans: {arch_score.four_iwans:.2f}</p>
    </div>
    <div class="card">
        <h3>Negative Control (Style Bleeding): {"LEAK DETECTED" if bleed_res.bleed_detected else "PASSED"}</h3>
        <p>{html.escape(bleed_res.notes)}</p>
    </div>
</body>
</html>"""
        out_p.write_text(html_code, encoding="utf-8")
        return out_p
