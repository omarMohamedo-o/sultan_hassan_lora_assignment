"""FLUX LoRA training launcher with MLflow versioning and GPU/CPU auto-detection.

Each training run produces a versioned model directory (never overwrites previous runs)
and logs all parameters, metrics, and artifacts to MLflow for full experiment tracking.
"""

import subprocess
import time
from pathlib import Path

import mlflow
import yaml

from sultan_hassan.config.loader import load_config
from sultan_hassan.utils import configure_utf8_streams


def resolve_device(configured_device: str) -> str:
    """Resolve the effective compute device.

    - "cuda": force GPU
    - "cpu": force CPU
    - "auto": detect GPU availability, fallback to CPU
    """
    if configured_device == "auto":
        try:
            import torch  # type: ignore

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return configured_device


def main() -> None:
    configure_utf8_streams()

    config = load_config()
    train_cfg = config.training
    device = resolve_device(config.project.device)

    # ── Versioned output directory (never overwrites) ──────────────────────
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    versioned_output = config.paths.models_lora / f"v_{timestamp}"
    versioned_output.mkdir(parents=True, exist_ok=True)

    training_manifest = {
        "base_model": train_cfg.base_model,
        "trigger_word": train_cfg.trigger_word,
        "lora_rank": train_cfg.lora_rank,
        "lora_alpha": train_cfg.lora_alpha,
        "learning_rate": train_cfg.learning_rate,
        "batch_size": train_cfg.batch_size,
        "gradient_accumulation_steps": train_cfg.gradient_accumulation_steps,
        "max_train_steps": train_cfg.max_train_steps,
        "epochs": train_cfg.epochs,
        "seed": train_cfg.seed,
        "precision": train_cfg.precision,
        "checkpoint_frequency": train_cfg.checkpoint_frequency,
        "training_data_dir": str(config.paths.final_images),
        "output_dir": str(versioned_output),
        "device": device,
    }

    # ── Write config snapshot ──────────────────────────────────────────────
    out_file = Path("deliverable/training/training_config.yaml")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(yaml.dump(training_manifest, default_flow_style=False), encoding="utf-8")

    # ── Build accelerate launch command ────────────────────────────────────
    accel_config = (
        "config/accelerate_cpu.yaml" if device == "cpu" else "config/accelerate_gpu.yaml"
    )
    mixed_precision = "no" if device == "cpu" else train_cfg.precision

    checkpoints_dir = Path("outputs/checkpoints")
    existing_checkpoints = (
        list(checkpoints_dir.glob("checkpoint-*")) if checkpoints_dir.exists() else []
    )
    resume_flag = ' --resume_from_checkpoint="latest"' if existing_checkpoints else ""

    # ── Dynamic Model Mapping ──────────────────────────────────────────────
    base_model = train_cfg.base_model.lower()
    if "flux" in base_model:
        train_script = "train_flux_lora.py"
        resolution = 1024
    elif "xl" in base_model:
        train_script = "train_sdxl_lora.py"
        resolution = 1024
    else:
        # Default to SD 1.5 logic for smaller/legacy models
        train_script = "train_sd15_lora.py"
        resolution = 512

    cmd = (
        f"accelerate launch --config_file={accel_config} {train_script} "
        f"--pretrained_model_name_or_path={train_cfg.base_model} "
        f"--instance_data_dir={config.paths.final_images} "
        f"--instance_prompt={train_cfg.trigger_word} "
        f"--output_dir={versioned_output} "
        f"--mixed_precision={mixed_precision} "
        f"--resolution={resolution} --train_batch_size={train_cfg.batch_size} "
        f"--gradient_accumulation_steps={train_cfg.gradient_accumulation_steps} "
        f"--checkpointing_steps={train_cfg.checkpoint_frequency} "
        f"--learning_rate={train_cfg.learning_rate} --rank={train_cfg.lora_rank} "
        f"--max_train_steps={train_cfg.max_train_steps} --seed={train_cfg.seed} "
        f"--gradient_checkpointing --use_8bit_adam"
        f"{resume_flag}"
    )

    cmd_file = Path("deliverable/training/training_command.txt")
    cmd_file.write_text(cmd, encoding="utf-8")

    print("=" * 72)
    print("SULTAN HASSAN LoRA – Training Launch")
    print("=" * 72)
    print(f"  Device           : {device.upper()}")
    print(f"  Accelerate Config: {accel_config}")
    print(f"  Model Version    : v_{timestamp}")
    print(f"  Output Dir       : {versioned_output}")
    print(f"  Config Snapshot  : {out_file}")
    print(f"  Command File     : {cmd_file}")
    print("=" * 72)

    # ── MLflow experiment tracking ─────────────────────────────────────────
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment(config.project.name)

    with mlflow.start_run(run_name=f"v_{timestamp}"):
        mlflow.log_params(training_manifest)
        mlflow.set_tag("device", device)
        mlflow.set_tag("version", f"v_{timestamp}")

        print("\n[INFO] Starting training via accelerate...")
        try:
            subprocess.run(cmd, shell=True, check=True)
            mlflow.set_tag("status", "success")

            # Log all model artifacts
            for f in versioned_output.iterdir():
                if f.is_file():
                    mlflow.log_artifact(str(f))

            # Register model in MLflow Model Registry
            model_uri = f"runs:/{mlflow.active_run().info.run_id}"  # type: ignore[union-attr]
            mlflow.register_model(model_uri, "sultan_hassan_flux_lora")

            print(f"\n[SUCCESS] Training completed. Model saved to {versioned_output}")
            print("[SUCCESS] Model registered in MLflow as 'sultan_hassan_flux_lora'")

        except subprocess.CalledProcessError as e:
            mlflow.set_tag("status", "failed")
            mlflow.log_metric("exit_code", e.returncode)
            print(f"\n[ERROR] Training failed with exit code {e.returncode}")
            raise


if __name__ == "__main__":
    main()
