"""Script to launch the local human review interface."""

import argparse

from sultan_hassan.config.loader import load_config
from sultan_hassan.review.reviewer import run_review_server
from sultan_hassan.utils import configure_utf8_streams


def main() -> None:
    configure_utf8_streams()

    parser = argparse.ArgumentParser(description="Run local human review web server")
    parser.add_argument(
        "--port", type=int, default=8080, help="Port to bind server (default: 8080)"
    )
    args = parser.parse_args()

    config = load_config()
    run_review_server(config, port=args.port)


if __name__ == "__main__":
    main()
