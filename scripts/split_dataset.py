"""CLI tool to split dataset into train, val, and test subsets based on config."""

import argparse
import sys

from sultan_hassan.config.loader import load_config
from sultan_hassan.dataset.manifest import DatasetManifestManager
from sultan_hassan.dataset.splitter import DatasetSplitter
from sultan_hassan.utils import configure_utf8_streams


def main() -> None:
    configure_utf8_streams()

    parser = argparse.ArgumentParser(
        description="Split final dataset into train, val, and test subsets"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=None,
        help="Explicit number of training images",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=None,
        help="Explicit number of validation images",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=None,
        help="Explicit number of test images",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    if args.train_count is not None:
        config.dataset.split.fixed_train_count = args.train_count
    if args.val_count is not None:
        config.dataset.split.fixed_val_count = args.val_count
    if args.test_count is not None:
        config.dataset.split.fixed_test_count = args.test_count

    manifest_mgr = DatasetManifestManager(config)
    records = manifest_mgr.load_records()

    if not records:
        print(
            "[WARNING] No records found in data/metadata/final_dataset.csv. Run prepare_dataset first."
        )
        sys.exit(0)

    splitter = DatasetSplitter(config)
    partitions = splitter.partition_records(records)
    summary = splitter.materialize_splits(partitions)

    print("[SUCCESS] Dataset partitioned according to config:")
    for split_name, count in summary.items():
        print(f"  - {split_name.upper()}: {count} images (in data/final/splits/{split_name}/)")


if __name__ == "__main__":
    main()
