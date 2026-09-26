"""CLI tool to frame images for training (letterbox, smart crop, aspect bucketing)."""

import argparse
import sys
from pathlib import Path

from sultan_hassan.config.loader import load_config
from sultan_hassan.images.framing import batch_frame_images, frame_image
from sultan_hassan.utils import configure_utf8_streams


def main() -> None:
    configure_utf8_streams()

    parser = argparse.ArgumentParser(description="Frame and format images for FLUX LoRA training")
    parser.add_argument(
        "--input",
        "-i",
        default="data/final/images",
        help="Input image file or directory (default: data/final/images)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/final/framed",
        help="Output directory (default: data/final/framed)",
    )
    parser.add_argument(
        "--mode",
        choices=["bucket", "smart_center_crop", "letterbox"],
        default=None,
        help="Override framing mode (default from config)",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config file",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    if args.mode:
        config.image.framing.mode = args.mode

    in_path = Path(args.input)
    out_dir = Path(args.output)

    if in_path.is_file():
        out_file = out_dir / in_path.name
        w, h = frame_image(in_path, out_file, config.image.framing)
        print(
            f"[SUCCESS] Framed {in_path.name} -> {out_file} ({w}x{h}, mode: {config.image.framing.mode})"
        )
    elif in_path.is_dir():
        framed = batch_frame_images(in_path, out_dir, config)
        print(
            f"[SUCCESS] Framed {len(framed)} images into {out_dir} using mode '{config.image.framing.mode}'."
        )
    else:
        print(f"[ERROR] Input path does not exist: {in_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
