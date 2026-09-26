"""CLI command pipeline runner."""

from sultan_hassan.config.loader import load_config
from sultan_hassan.pipeline.stages import (
    run_collect_stage,
    run_dataset_stage,
    run_deliver_stage,
    run_evaluate_stage,
    run_generate_stage,
    run_review_stage,
    run_standardize_stage,
    run_train_stage,
)
from sultan_hassan.utils import configure_utf8_streams


def run_pipeline(stage: str = "collect", config_path: str = "config/config.yaml") -> int:
    """Run specified pipeline stage."""
    configure_utf8_streams()

    config = load_config(config_path)

    if stage == "standardize":
        run_standardize_stage(config, cleanup=True)
        return 0
    elif stage == "collect":
        run_collect_stage(config)
        return 0
    elif stage == "review":
        run_review_stage(config)
        return 0
    elif stage == "dataset":
        run_dataset_stage(config)
        return 0
    elif stage == "train":
        run_train_stage(config)
        return 0
    elif stage == "generate":
        run_generate_stage(config)
        return 0
    elif stage == "evaluate":
        run_evaluate_stage(config)
        return 0
    elif stage == "deliver":
        run_deliver_stage(config)
        return 0
    else:
        print(f"[ERROR] Unknown pipeline stage: '{stage}'")
        return 1
