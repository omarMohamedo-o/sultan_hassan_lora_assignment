"""Script to discover and inspect raw video assets."""

from rich.console import Console
from rich.table import Table

from sultan_hassan.config.loader import load_config
from sultan_hassan.utils import configure_utf8_streams
from sultan_hassan.video.inspector import VideoInspector

console = Console()


def main() -> None:
    configure_utf8_streams()

    config = load_config()
    inspector = VideoInspector(config)
    videos = inspector.inspect_all()

    summary = inspector.get_summary(videos)
    table = Table(title="Video Inspection Report", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold green")

    for k, v in summary.items():
        table.add_row(k.replace("_", " ").title(), str(v))

    console.print(table)


if __name__ == "__main__":
    main()
