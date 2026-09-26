"""Script to define test prompts, generate fixed-seed test images, and build benchmark gallery."""

import html
import shutil
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from sultan_hassan.config.loader import load_config
from sultan_hassan.generation.generator import FluxGenerator, GenerationParams
from sultan_hassan.utils import configure_utf8_streams

TEST_PROMPTS = [
    {
        "group": 1,
        "name": "Monumental Elevation",
        "prompt": "sltnhsn, elevation",
        "has_trigger": True,
        "description": "Massive plain limestone walls, vertical recessed window bays, and minarets.",
    },
    {
        "group": 2,
        "name": "Monumental Portal & Gates",
        "prompt": "sltnhsn, entrance gates",
        "has_trigger": True,
        "description": "Monumental projecting portal, tiered muqarnas stalactite corbelling, and ablaq masonry.",
    },
    {
        "group": 3,
        "name": "Courtyard & Central Fountain",
        "prompt": "sltnhsn, a courtyard with a central fountain at sunset",
        "has_trigger": True,
        "description": "4-iwan cruciform courtyard, octagonal ablution fountain with carved dome, golden sunset.",
    },
    {
        "group": 4,
        "name": "Mosque Nocturnal Atmosphere",
        "prompt": "sltnhsn, a mosque at night",
        "has_trigger": True,
        "description": "Midnight sky, dramatic architectural uplighting on minaret balconies, and glowing lamps.",
    },
    {
        "group": 5,
        "name": "Negative Control (Office Tower)",
        "prompt": "a modern glass office tower",
        "has_trigger": False,
        "description": "Modern corporate glass curtain-wall grid; verifies zero style-bleeding or trigger leakage.",
    },
]

SEEDS = [101, 202, 303, 404]


def build_test_gallery_html(records: list[dict[str, object]], out_path: Path) -> None:
    """Generate modern dark-mode responsive HTML gallery for the test outputs."""
    groups_html = ""
    for group_def in TEST_PROMPTS:
        gid = group_def["group"]
        group_records = [r for r in records if r["group"] == gid]
        cards_html = ""
        for r in group_records:
            img_rel = Path(str(r["image_path"])).name
            cards_html += f"""
            <div class="card">
                <img src="{img_rel}" alt="{html.escape(str(r["sample_id"]))}" loading="lazy" />
                <div class="card-info">
                    <span class="badge seed">Seed: {r["seed"]}</span>
                    <span class="badge {"trigger" if r["has_trigger"] else "control"}">
                        {"Trigger Active" if r["has_trigger"] else "Negative Control"}
                    </span>
                    <div class="prompt">"{html.escape(str(r["prompt"]))}"</div>
                    <div class="meta">Steps: {r["steps"]} | Guidance: {r["guidance"]} | Res: {r["resolution"]}</div>
                </div>
            </div>
            """

        groups_html += f"""
        <section class="group-section">
            <div class="group-header">
                <h2>Group {gid}: {html.escape(str(group_def["name"]))}</h2>
                <p class="group-desc">{html.escape(str(group_def["description"]))}</p>
            </div>
            <div class="cards-grid">
                {cards_html}
            </div>
        </section>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sultan Hassan FLUX LoRA - Benchmark Test Gallery</title>
    <style>
        :root {{
            --bg: #0d1117;
            --surface: #161b22;
            --border: #30363d;
            --text: #c9d1d9;
            --text-heading: #f0f6fc;
            --accent: #58a6ff;
            --success: #3fb950;
            --warning: #d29922;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            padding: 32px 24px;
            line-height: 1.5;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        header {{ margin-bottom: 40px; border-bottom: 1px solid var(--border); padding-bottom: 24px; }}
        h1 {{ font-size: 2.2rem; color: var(--text-heading); margin-bottom: 8px; }}
        .subtitle {{ font-size: 1.05rem; color: #8b949e; }}
        .group-section {{ margin-bottom: 48px; }}
        .group-header {{ margin-bottom: 20px; }}
        .group-header h2 {{ font-size: 1.4rem; color: var(--accent); margin-bottom: 4px; }}
        .group-desc {{ color: #8b949e; font-size: 0.95rem; }}
        .cards-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 24px;
        }}
        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}
        .card:hover {{ transform: translateY(-4px); border-color: var(--accent); }}
        .card img {{ width: 100%; height: auto; display: block; aspect-ratio: 1/1; object-fit: cover; }}
        .card-info {{ padding: 16px; }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 6px;
            margin-bottom: 8px;
        }}
        .badge.seed {{ background: #21262d; color: #58a6ff; border: 1px solid #30363d; }}
        .badge.trigger {{ background: rgba(63, 185, 80, 0.15); color: #3fb950; border: 1px solid rgba(63, 185, 80, 0.4); }}
        .badge.control {{ background: rgba(210, 153, 34, 0.15); color: #d29922; border: 1px solid rgba(210, 153, 34, 0.4); }}
        .prompt {{ font-weight: 600; color: var(--text-heading); font-size: 0.95rem; margin-bottom: 8px; }}
        .meta {{ font-size: 0.8rem; color: #8b949e; font-family: monospace; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Sultan Hassan FLUX LoRA - Benchmark Test Suite</h1>
            <p class="subtitle">20 Test Images across 5 Benchmark Prompt Groups with Fixed Reproducible Seeds (101, 202, 303, 404)</p>
        </header>
        {groups_html}
    </div>
</body>
</html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content, encoding="utf-8")


def main() -> None:
    configure_utf8_streams()
    console = Console()

    config = load_config()
    out_dir = Path(config.paths.outputs_tests)
    out_dir.mkdir(parents=True, exist_ok=True)

    deliv_tests = Path("deliverable/tests")
    deliv_tests.mkdir(parents=True, exist_ok=True)

    generator = FluxGenerator()
    records: list[dict[str, object]] = []

    table = Table(title="Generating FLUX LoRA Test Benchmark Suite")
    table.add_column("Group", style="cyan", justify="right")
    table.add_column("Seed", style="yellow", justify="right")
    table.add_column("Prompt", style="white")
    table.add_column("File", style="green")

    for p in TEST_PROMPTS:
        for seed_idx, seed in enumerate(SEEDS, start=1):
            sample_id = f"group{p['group']}_sample{seed_idx}_seed{seed}"
            img_filename = f"{sample_id}.jpg"
            img_path = out_dir / img_filename

            params = GenerationParams(
                prompt=str(p["prompt"]),
                seed=seed,
                steps=28,
                guidance=3.5,
                width=1024,
                height=1024,
                group_id=int(p["group"]),
            )

            generator.generate(params, img_path)
            shutil.copy(img_path, deliv_tests / img_filename)

            record = {
                "sample_id": sample_id,
                "group": p["group"],
                "group_name": p["name"],
                "prompt": p["prompt"],
                "seed": seed,
                "has_trigger": p["has_trigger"],
                "steps": 28,
                "guidance": 3.5,
                "resolution": "1024x1024",
                "image_path": str(img_path),
            }
            records.append(record)
            table.add_row(str(p["group"]), str(seed), str(p["prompt"]), img_filename)

    console.print(table)

    # Save CSV metadata
    csv_paths = [
        out_dir / "test_metadata.csv",
        Path("deliverable/metadata/test_metadata.csv"),
    ]
    df = pd.DataFrame(records)
    for cp in csv_paths:
        cp.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cp, index=False)

    # Save HTML gallery
    gallery_paths = [
        out_dir / "index.html",
        Path("deliverable/reports/test_gallery.html"),
    ]
    for gp in gallery_paths:
        build_test_gallery_html(records, gp)

    console.print(
        f"[bold green]Successfully generated {len(records)} test images in: {out_dir}[/bold green]"
    )
    console.print(
        f"[bold green]Interactive test gallery created in: {gallery_paths[0]}[/bold green]"
    )


if __name__ == "__main__":
    main()
