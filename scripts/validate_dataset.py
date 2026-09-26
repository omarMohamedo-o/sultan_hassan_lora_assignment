"""Script to execute strict dataset validation quality gates."""

import sys

from sultan_hassan.config.loader import load_config
from sultan_hassan.dataset.validator import DatasetValidator
from sultan_hassan.utils import configure_utf8_streams


def main() -> None:
    configure_utf8_streams()

    config = load_config()
    validator = DatasetValidator(config)
    res = validator.validate(raise_on_failure=False)

    print(f"Validation Status: {'PASSED' if res.is_valid else 'FAILED'}")
    print(f"Total Images: {res.total_images}")
    print(f"Errors: {res.error_count}")
    print(f"Warnings: {res.warning_count}")

    if not res.is_valid:
        for err in res.errors:
            print(f"  - ERROR: {err}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
