"""Script to generate visual contact sheets and HTML gallery."""

from pathlib import Path

from sultan_hassan.config.loader import load_config
from sultan_hassan.review.contact_sheet import ContactSheetGenerator, create_thumbnail_card
from sultan_hassan.utils import configure_utf8_streams


def main() -> None:
    configure_utf8_streams()

    config = load_config()
    candidates_dir = Path(config.paths.candidates_images)
    frames = sorted(candidates_dir.glob("*.jpg"))

    generator = ContactSheetGenerator(config)
    cards = []
    items = []

    for f in frames:
        stem_parts = f.stem.split("_")
        frame_id = "_".join(stem_parts[-2:]) if len(stem_parts) >= 2 else f.stem
        card = create_thumbnail_card(
            image_path=f,
            frame_id=frame_id,
            source_video=f.name,
            timestamp_sec=0.0,
            resolution="1024x1024",
            quality_score=75.0,
            sultan_score=0.9,
            rifai_score=0.1,
            status="KEEP",
        )
        cards.append(card)
        items.append(
            {
                "image_path": f,
                "frame_id": frame_id,
                "source_video": f.name,
                "resolution": "1024x1024",
                "timestamp_sec": 0.0,
                "quality_score": 75.0,
                "blur_score": 120.0,
                "sultan_score": 0.9,
                "rifai_score": 0.1,
                "status": "KEEP",
            }
        )

    img_out = Path(config.paths.reports_contact_sheets) / "candidates_contact_sheet.jpg"
    html_out = Path(config.paths.reports_contact_sheets) / "index.html"

    generator.generate_grid_image(cards, img_out)
    generator.generate_html_gallery(items, "Candidate Inspection Contact Sheet", html_out)
    print(f"[SUCCESS] Contact sheet image saved: {img_out}")
    print(f"[SUCCESS] Interactive HTML gallery saved: {html_out}")


if __name__ == "__main__":
    main()
