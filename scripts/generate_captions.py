"""Script to generate truthful architectural captions with trigger word sltnhsn."""

from pathlib import Path

from sultan_hassan.config.loader import load_config
from sultan_hassan.dataset.captions import (
    CaptionManager,
    generate_caption_for_category,
    validate_caption,
)
from sultan_hassan.domain.enums import CategoryEnum
from sultan_hassan.utils import configure_utf8_streams


def main() -> None:
    configure_utf8_streams()

    config = load_config()
    mgr = CaptionManager(config)
    final_imgs = sorted(Path(config.paths.final_images).glob("*.jpg"))

    count = 0
    for img_p in final_imgs:
        caption = generate_caption_for_category(
            CategoryEnum.EXTERIOR, trigger_word=config.dataset.trigger_word
        )
        is_val, err, _ = validate_caption(
            caption, trigger_word=config.dataset.trigger_word, filename=img_p.name
        )
        if is_val:
            mgr.write_caption(img_p.stem, caption)
            count += 1

    print(f"[SUCCESS] Generated and verified {count} captions in {config.paths.final_captions}")


if __name__ == "__main__":
    main()
