"""CLI tool to standardize all raw and external images/videos to high quality .jpg and .mp4."""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from sultan_hassan.images.standardizer import (
    standardize_image_to_high_quality_jpg,
    standardize_media_directory,
    standardize_video_to_high_quality_mp4,
    verify_image_standard_quality,
    verify_video_standard_quality,
)
from sultan_hassan.utils import configure_utf8_streams


def main() -> None:
    configure_utf8_streams()
    console = Console()

    parser = argparse.ArgumentParser(
        description="Standardize images and videos to pristine standard formats (.jpg and .mp4)"
    )
    parser.add_argument(
        "--path",
        default="data/raw",
        help="Target file or directory to standardize (default: data/raw)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output directory. Defaults to standardizing in-place or matching directory.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=96,
        help="JPEG quality for standardized images (default: 96, 4:4:4 subsampling)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove original non-standard files (.heic, .webp, .mov, etc.) after successful conversion",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive subfolder processing",
    )

    args = parser.parse_args()
    target = Path(args.path)

    if target.is_dir():
        results = standardize_media_directory(
            target,
            output_directory=args.output,
            quality=args.quality,
            cleanup_non_standard=args.cleanup,
            recursive=not args.no_recursive,
        )

        table = Table(title="Media Standardization & Quality Verification Summary")
        table.add_column("Media Category", style="cyan", no_wrap=True)
        table.add_column("Count", style="green", justify="right")
        table.add_column("Standard Format", style="yellow")
        table.add_column("Quality Status", style="magenta")

        table.add_row(
            "Images Standardized",
            str(len(results["images"])),
            ".jpg (RGB 4:4:4)",
            f"Quality {args.quality} verified",
        )
        table.add_row(
            "Videos Standardized",
            str(len(results["videos"])),
            ".mp4 (H.264/AVC)",
            "Container & stream verified",
        )

        console.print(table)
        console.print(
            f"[bold green]Standardization completed successfully for directory: {target}[/bold green]"
        )

    elif target.is_file():
        if target.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv"]:
            out_file = Path(args.output) if args.output else target.with_suffix(".mp4")
            success = standardize_video_to_high_quality_mp4(target, out_file)
            if success:
                valid, msg = verify_video_standard_quality(out_file)
                if valid:
                    console.print(
                        f"[bold green][SUCCESS][/bold green] Video standardized: {out_file} ({msg})"
                    )
                    if args.cleanup and target != out_file:
                        target.unlink()
                        console.print(f"[dim]Cleaned up source file: {target.name}[/dim]")
                else:
                    console.print(f"[bold red][ERROR][/bold red] Video verification failed: {msg}")
                    sys.exit(1)
            else:
                console.print(
                    f"[bold red][ERROR][/bold red] Video standardization failed for: {target}"
                )
                sys.exit(1)
        else:
            out_file = Path(args.output) if args.output else target.with_suffix(".jpg")
            w, h = standardize_image_to_high_quality_jpg(target, out_file, quality=args.quality)
            valid, msg = verify_image_standard_quality(out_file)
            if valid:
                console.print(
                    f"[bold green][SUCCESS][/bold green] Image standardized: {out_file} ({w}x{h}, quality={args.quality})"
                )
                if args.cleanup and target != out_file:
                    target.unlink()
                    console.print(f"[dim]Cleaned up source file: {target.name}[/dim]")
            else:
                console.print(f"[bold red][ERROR][/bold red] Image verification failed: {msg}")
                sys.exit(1)
    else:
        console.print(f"[bold red][ERROR][/bold red] Path not found: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()
