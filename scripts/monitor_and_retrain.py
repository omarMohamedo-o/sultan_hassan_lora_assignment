"""CLI tool to monitor model evaluation metrics and trigger retraining on performance drop."""

import argparse
import subprocess

from sultan_hassan.evaluation.metrics import ArchitecturalScore, StyleBleedingResult
from sultan_hassan.training.drift_monitor import ModelDriftMonitor
from sultan_hassan.utils import configure_utf8_streams


def main() -> None:
    configure_utf8_streams()

    parser = argparse.ArgumentParser(
        description="Monitor model evaluation scores and trigger Docker retraining on drift"
    )
    parser.add_argument(
        "--min-acc",
        type=float,
        default=0.85,
        help="Minimum acceptable accuracy before retraining (default: 0.85)",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Launch training inside Docker container (docker-compose)",
    )

    args = parser.parse_args()

    monitor = ModelDriftMonitor(min_acceptable_accuracy=args.min_acc)

    # In production, these scores are derived from test generation evaluation
    current_arch = ArchitecturalScore(overall_accuracy=0.92)
    current_bleed = StyleBleedingResult(bleed_score=0.03)

    status = monitor.evaluate_drift(current_arch, current_bleed)

    print("\n--- Drift Monitor Status ---")
    print(f"Accuracy Score: {status.accuracy_score}")
    print(f"Style Bleeding Score: {status.style_bleed_score}")
    print(f"Drift Detected: {status.drift_detected}")
    print(f"Retrain Recommended: {status.retrain_recommended}")
    print(f"Status: {status.rationale}\n")

    if status.retrain_recommended:
        print("[ACTION] Launching retraining pipeline to remediate drift...")
        if args.docker:
            cmd = ["docker", "compose", "-f", "docker-compose.train.yml", "up", "--build"]
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd)
        else:
            cmd = ["python", "scripts/train_lora.py"]
            subprocess.run(cmd)


if __name__ == "__main__":
    main()
