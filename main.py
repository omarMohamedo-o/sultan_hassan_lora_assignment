"""Main CLI entrypoint for Sultan Hassan ML Pipeline."""

import argparse
import sys

from sultan_hassan.pipeline.runner import run_pipeline
from sultan_hassan.utils import configure_utf8_streams


def main() -> None:
    configure_utf8_streams()

    parser = argparse.ArgumentParser(
        description="Sultan Hassan Mosque ML Pipeline & FLUX LoRA Curation"
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="collect",
        choices=[
            "standardize",
            "collect",
            "filter",
            "review",
            "dataset",
            "train",
            "generate",
            "evaluate",
            "deliver",
        ],
        help="Pipeline stage to execute (default: collect)",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML configuration file (default: config/config.yaml)",
    )

    args = parser.parse_args()
    code = run_pipeline(stage=args.stage, config_path=args.config)
    sys.exit(code)


if __name__ == "__main__":
    main()
