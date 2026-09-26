"""Visual contact sheet and interactive HTML review gallery generator."""

import html
import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from sultan_hassan.config.models import AppConfig

logger = logging.getLogger(__name__)


def create_thumbnail_card(
    image_path: Path,
    frame_id: str,
    source_video: str,
    timestamp_sec: float,
    resolution: str,
    quality_score: float,
    sultan_score: float,
    rifai_score: float,
    status: str,
    thumb_width: int = 320,
    thumb_height: int = 240,
    card_height: int = 340,
) -> Image.Image:
    """Create a composite image thumbnail card with annotated metadata banner."""
    card = Image.new("RGB", (thumb_width, card_height), color=(28, 30, 36))

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
            # Center thumbnail on upper card area
            offset_x = (thumb_width - img.width) // 2
            offset_y = (thumb_height - img.height) // 2
            card.paste(img, (offset_x, offset_y))
    except Exception as exc:
        logger.warning("Could not render thumbnail for %s: %s", image_path.name, exc)

    draw = ImageDraw.Draw(card)
    # Default PIL bitmap font
    font = ImageFont.load_default()

    # Status color badge
    status_color = (
        (40, 180, 99)
        if status == "KEEP"
        else ((243, 156, 18) if status == "REVIEW" else (231, 76, 60))
    )

    banner_y = thumb_height + 4
    draw.rectangle(
        [(4, banner_y), (thumb_width - 4, card_height - 4)], fill=(18, 20, 24), outline=(50, 54, 62)
    )

    # Text metadata
    draw.text((10, banner_y + 4), f"ID: {frame_id[:16]}", fill=(255, 255, 255), font=font)
    draw.text(
        (10, banner_y + 18),
        f"Time: {timestamp_sec:.2f}s | Res: {resolution}",
        fill=(200, 200, 200),
        font=font,
    )
    draw.text(
        (10, banner_y + 32),
        f"Q-Score: {quality_score:.1f} | Status: {status}",
        fill=status_color,
        font=font,
    )
    draw.text(
        (10, banner_y + 46),
        f"Sultan: {sultan_score:.2f} | Rifai: {rifai_score:.2f}",
        fill=(160, 210, 255),
        font=font,
    )
    short_src = source_video[:30] + ("..." if len(source_video) > 30 else "")
    draw.text((10, banner_y + 60), f"Src: {short_src}", fill=(140, 145, 155), font=font)

    return card


class ContactSheetGenerator:
    """Generates visual grid contact sheets and responsive HTML inspection reports."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.reports_dir = Path(config.paths.reports_contact_sheets)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_grid_image(
        self,
        cards: list[Image.Image],
        output_path: Path,
        cols: int = 4,
    ) -> Path:
        """Compose individual thumbnail cards into a high-resolution grid image."""
        if not cards:
            return output_path

        card_w, card_h = cards[0].size
        rows = (len(cards) + cols - 1) // cols
        grid_w = cols * (card_w + 12) + 12
        grid_h = rows * (card_h + 12) + 12

        grid = Image.new("RGB", (grid_w, grid_h), color=(14, 16, 20))
        for idx, card in enumerate(cards):
            r = idx // cols
            c = idx % cols
            x = 12 + c * (card_w + 12)
            y = 12 + r * (card_h + 12)
            grid.paste(card, (x, y))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        grid.save(output_path, "JPEG", quality=90)
        return output_path

    def generate_html_gallery(
        self,
        items: list[dict[str, object]],
        title: str,
        output_path: Path,
    ) -> Path:
        """Generate a sleek, responsive dark-mode HTML inspection gallery."""
        html_cards = []
        for it in items:
            status = str(it.get("status", "REVIEW"))
            badge_class = (
                "keep" if status == "KEEP" else ("review" if status == "REVIEW" else "reject")
            )
            src_path = Path(str(it.get("image_path", "")))
            rel_src = src_path.as_posix()

            html_cards.append(f"""
            <div class="card">
                <div class="img-wrap">
                    <img src="{rel_src}" alt="{html.escape(str(it.get("frame_id")))}" loading="lazy" />
                    <span class="badge {badge_class}">{status}</span>
                </div>
                <div class="meta">
                    <div class="id"><strong>ID:</strong> {html.escape(str(it.get("frame_id")))}</div>
                    <div><strong>Resolution:</strong> {html.escape(str(it.get("resolution")))}</div>
                    <div><strong>Timestamp:</strong> {it.get("timestamp_sec", 0.0):.2f}s</div>
                    <div><strong>Quality:</strong> {it.get("quality_score", 0.0):.1f} (Blur: {it.get("blur_score", 0.0):.1f})</div>
                    <div><strong>Sultan Hassan:</strong> {it.get("sultan_score", 0.0):.2f} | <strong>Al-Rifa'i:</strong> {it.get("rifai_score", 0.0):.2f}</div>
                    <div class="src" title="{html.escape(str(it.get("source_video")))}"><strong>Source:</strong> {html.escape(str(it.get("source_video")))}</div>
                </div>
            </div>
            """)

        cards_str = "\n".join(html_cards)
        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)} - Sultan Hassan ML Pipeline</title>
    <style>
        :root {{
            --bg: #0f1117;
            --surface: #181b22;
            --surface-hover: #212631;
            --border: #2d3340;
            --text: #e6edf3;
            --text-muted: #8b949e;
            --accent: #58a6ff;
            --green: #238636;
            --yellow: #d29922;
            --red: #da3633;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            padding: 24px;
        }}
        header {{
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        h1 {{ font-size: 1.5rem; font-weight: 600; color: #fff; }}
        .count {{ color: var(--text-muted); font-size: 0.95rem; }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            transition: transform 0.15s ease, border-color 0.15s ease;
        }}
        .card:hover {{
            transform: translateY(-2px);
            border-color: var(--accent);
        }}
        .img-wrap {{
            position: relative;
            background: #000;
            height: 200px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .img-wrap img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}
        .badge {{
            position: absolute;
            top: 8px;
            right: 8px;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .badge.keep {{ background: var(--green); color: #fff; }}
        .badge.review {{ background: var(--yellow); color: #111; }}
        .badge.reject {{ background: var(--red); color: #fff; }}
        .meta {{
            padding: 12px;
            font-size: 0.85rem;
            line-height: 1.5;
            color: var(--text-muted);
        }}
        .meta strong {{ color: var(--text); }}
        .meta .id {{ font-family: monospace; color: var(--accent); margin-bottom: 4px; }}
        .meta .src {{
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin-top: 4px;
            font-size: 0.75rem;
        }}
    </style>
</head>
<body>
    <header>
        <h1>{html.escape(title)}</h1>
        <div class="count">{len(items)} items inspected</div>
    </header>
    <div class="grid">
        {cards_str}
    </div>
</body>
</html>
"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(full_html, encoding="utf-8")
        return output_path
