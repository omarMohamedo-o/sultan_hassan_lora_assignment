"""Export trained LoRA weights to ONNX format for hardware-accelerated inference.

Produces an ONNX model that can be consumed by:
- Python onnxruntime (DirectML / CUDA / TensorRT / OpenVINO / CPU)
- C++ onnxruntime binary
- Any ONNX-compatible runtime

Each export is versioned and logged to MLflow.
"""

import shutil
import time
from pathlib import Path

import mlflow
import numpy as np
import onnx
import yaml
from onnx import TensorProto, helper
from safetensors.numpy import load_file, save_file

from sultan_hassan.config.loader import load_config
from sultan_hassan.utils import configure_utf8_streams


def find_latest_lora_version(models_dir: Path) -> Path | None:
    """Find the most recent versioned LoRA training output."""
    versions = sorted(models_dir.glob("v_*"), reverse=True)
    for v in versions:
        safetensors = list(v.glob("*.safetensors"))
        if safetensors:
            return safetensors[0]
    return None


def create_lora_projection_onnx(
    lora_rank: int,
    lora_alpha: int,
    hidden_dim: int = 3072,
    output_path: Path | None = None,
) -> onnx.ModelProto:
    """Build an ONNX graph representing the LoRA low-rank projection.

    Architecture:  input -> LoRA_Down (hidden_dim x rank) -> LoRA_Up (rank x hidden_dim) -> scaled output
    This mirrors the actual LoRA delta:  delta_W = (alpha/rank) * B @ A
    """
    scale = float(lora_alpha) / float(lora_rank)

    # Input tensor
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, hidden_dim])

    # LoRA Down projection weights (hidden_dim -> rank)
    lora_down_init = np.random.randn(hidden_dim, lora_rank).astype(np.float32) * 0.01
    lora_down_tensor = helper.make_tensor(
        "lora_down_weight", TensorProto.FLOAT, [hidden_dim, lora_rank], lora_down_init.flatten()
    )

    # LoRA Up projection weights (rank -> hidden_dim)
    lora_up_init = np.zeros((lora_rank, hidden_dim), dtype=np.float32)
    lora_up_tensor = helper.make_tensor(
        "lora_up_weight", TensorProto.FLOAT, [lora_rank, hidden_dim], lora_up_init.flatten()
    )

    # Scale constant
    scale_tensor = helper.make_tensor(
        "lora_scale", TensorProto.FLOAT, [], [scale]
    )

    # Graph nodes
    matmul_down = helper.make_node("MatMul", ["input", "lora_down_weight"], ["down_proj"])
    matmul_up = helper.make_node("MatMul", ["down_proj", "lora_up_weight"], ["up_proj"])
    mul_scale = helper.make_node("Mul", ["up_proj", "lora_scale"], ["output"])

    # Output tensor
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, hidden_dim])

    # Build graph
    graph = helper.make_graph(
        [matmul_down, matmul_up, mul_scale],
        "sultan_hassan_lora_projection",
        [X],
        [Y],
        initializer=[lora_down_tensor, lora_up_tensor, scale_tensor],
    )

    # Build model with opset 17
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.doc_string = (
        "Sultan Hassan Mosque-Madrasa FLUX LoRA projection. "
        f"Rank={lora_rank}, Alpha={lora_alpha}, Scale={scale:.4f}"
    )
    model.model_version = 1

    # Validate
    onnx.checker.check_model(model)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(output_path))

    return model


def inject_safetensors_weights(
    onnx_path: Path,
    safetensors_path: Path,
    output_path: Path,
) -> None:
    """Load trained LoRA weights from safetensors and inject into ONNX model."""
    model = onnx.load(str(onnx_path))
    weights = load_file(str(safetensors_path))

    # Map safetensors keys to ONNX initializer names
    weight_map = {}
    for key, tensor in weights.items():
        if "lora_down" in key or "lora_A" in key:
            weight_map["lora_down_weight"] = tensor
        elif "lora_up" in key or "lora_B" in key:
            weight_map["lora_up_weight"] = tensor

    # Replace initializers
    for init in model.graph.initializer:
        if init.name in weight_map:
            w = weight_map[init.name].astype(np.float32)
            new_init = helper.make_tensor(
                init.name, TensorProto.FLOAT, list(w.shape), w.flatten()
            )
            init.CopyFrom(new_init)

    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    print(f"[OK] Injected trained weights -> {output_path}")


def main() -> None:
    configure_utf8_streams()

    config = load_config()
    train_cfg = config.training
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    onnx_dir = Path("models/onnx")
    onnx_dir.mkdir(parents=True, exist_ok=True)

    versioned_onnx = onnx_dir / f"v_{timestamp}"
    versioned_onnx.mkdir(parents=True, exist_ok=True)

    onnx_path = versioned_onnx / "sltnhsn_flux_lora.onnx"

    print("=" * 72)
    print("SULTAN HASSAN LoRA – ONNX Export")
    print("=" * 72)

    # Step 1: Create base ONNX model
    print("\n[1/3] Building ONNX LoRA projection graph...")
    model = create_lora_projection_onnx(
        lora_rank=train_cfg.lora_rank,
        lora_alpha=train_cfg.lora_alpha,
        output_path=onnx_path,
    )
    print(f"  Base ONNX model saved: {onnx_path}")
    print(f"  Opset version: {model.opset_import[0].version}")

    # Step 2: Try to inject trained weights
    print("\n[2/3] Looking for trained safetensors weights...")
    latest_safetensors = find_latest_lora_version(config.paths.models_lora)
    if latest_safetensors:
        print(f"  Found: {latest_safetensors}")
        trained_onnx = versioned_onnx / "sltnhsn_flux_lora_trained.onnx"
        inject_safetensors_weights(onnx_path, latest_safetensors, trained_onnx)
        final_onnx = trained_onnx
    else:
        print("  No trained weights found. Exporting base (untrained) ONNX model.")
        final_onnx = onnx_path

    # Step 3: Copy to deliverable
    print("\n[3/3] Copying to deliverable/model/...")
    deliverable_model = Path("deliverable/model")
    deliverable_model.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_onnx, deliverable_model / "sltnhsn_flux_lora.onnx")

    # Save export metadata
    meta = {
        "format": "ONNX",
        "opset": 17,
        "lora_rank": train_cfg.lora_rank,
        "lora_alpha": train_cfg.lora_alpha,
        "exported_at": timestamp,
        "source_safetensors": str(latest_safetensors) if latest_safetensors else None,
        "compatible_runtimes": [
            "onnxruntime (Python)",
            "onnxruntime (C++/C#)",
            "DirectML",
            "CUDA EP",
            "TensorRT EP",
            "OpenVINO EP",
        ],
    }
    meta_path = versioned_onnx / "export_metadata.yaml"
    meta_path.write_text(yaml.dump(meta, default_flow_style=False), encoding="utf-8")

    # MLflow logging
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment(config.project.name)
    with mlflow.start_run(run_name=f"onnx_export_{timestamp}"):
        mlflow.log_params({
            "export_format": "ONNX",
            "opset_version": 17,
            "lora_rank": train_cfg.lora_rank,
            "lora_alpha": train_cfg.lora_alpha,
        })
        mlflow.log_artifact(str(final_onnx))
        mlflow.set_tag("stage", "onnx_export")
        mlflow.set_tag("version", f"v_{timestamp}")

    print(f"\n{'=' * 72}")
    print(f"[SUCCESS] ONNX model exported to: {final_onnx}")
    print(f"[SUCCESS] Deliverable copy: deliverable/model/sltnhsn_flux_lora.onnx")
    print(f"[SUCCESS] Logged to MLflow as onnx_export_{timestamp}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
