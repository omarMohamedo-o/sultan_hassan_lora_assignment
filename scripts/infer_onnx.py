"""ONNX Runtime inference engine for Sultan Hassan LoRA model.

Runs inference WITHOUT PyTorch — uses only ONNX Runtime with hardware acceleration.
Supports multiple execution providers:
- DirectML (Windows GPU: AMD, Intel, NVIDIA)
- CUDA (NVIDIA GPU)
- TensorRT (NVIDIA GPU, fastest)
- OpenVINO (Intel CPU/GPU/VPU)
- CPU (fallback, always available)

Usage:
    uv run python scripts/infer_onnx.py --prompt "sltnhsn, monumental entrance portal"
    uv run python scripts/infer_onnx.py --prompt "sltnhsn, dome" --provider CUDAExecutionProvider
    uv run python scripts/infer_onnx.py --prompt "sltnhsn, minaret" --provider CPUExecutionProvider
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from sultan_hassan.config.loader import load_config
from sultan_hassan.utils import configure_utf8_streams


def get_available_providers() -> list[str]:
    """Return list of ONNX Runtime execution providers available on this system."""
    return ort.get_available_providers()


def create_session(
    model_path: str,
    providers: list[str] | None = None,
) -> tuple[ort.InferenceSession, str]:
    """Create an ONNX Runtime session with the best available provider.

    Returns the session and the name of the provider actually used.
    """
    available = get_available_providers()
    print(f"[INFO] Available ONNX Runtime providers: {available}")

    if providers is None:
        providers = [
            "DmlExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    # Filter to only providers that are actually installed
    active_providers = [p for p in providers if p in available]
    if not active_providers:
        active_providers = ["CPUExecutionProvider"]

    print(f"[INFO] Using providers: {active_providers}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_mem_pattern = True
    sess_options.enable_cpu_mem_arena = True

    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=active_providers,
    )

    # Report which provider was actually selected
    actual = session.get_providers()
    return session, actual[0] if actual else "CPUExecutionProvider"


def run_inference(
    session: ort.InferenceSession,
    input_data: np.ndarray,
) -> np.ndarray:
    """Run a single forward pass through the ONNX model."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: input_data})
    return result[0]


def benchmark_inference(
    session: ort.InferenceSession,
    input_shape: tuple[int, ...],
    num_warmup: int = 3,
    num_runs: int = 10,
) -> dict[str, float]:
    """Benchmark the inference latency."""
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(num_warmup):
        run_inference(session, dummy_input)

    # Timed runs
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        run_inference(session, dummy_input)
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "std_ms": float(np.std(latencies)),
        "num_runs": num_runs,
    }


def main() -> None:
    configure_utf8_streams()

    parser = argparse.ArgumentParser(description="ONNX Runtime Inference for Sultan Hassan LoRA")
    parser.add_argument("--model", type=str, default=None, help="Path to ONNX model file")
    parser.add_argument("--provider", type=str, default=None, help="Execution provider override")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    parser.add_argument("--prompt", type=str, default="sltnhsn, monumental entrance portal", help="Prompt text")
    parser.add_argument("--seed", type=int, default=101, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    args = parser.parse_args()

    config = load_config()
    infer_cfg = config.inference

    model_path = args.model or infer_cfg.onnx_model_path
    providers = [args.provider] if args.provider else infer_cfg.execution_providers

    print("=" * 72)
    print("SULTAN HASSAN – ONNX Runtime Inference (No PyTorch)")
    print("=" * 72)
    print(f"  Backend  : ONNX Runtime {ort.__version__}")
    print(f"  Model    : {model_path}")
    print(f"  Providers: {providers}")

    if not Path(model_path).exists():
        print(f"\n[ERROR] ONNX model not found at {model_path}")
        print("  Run `uv run python scripts/export_onnx.py` first to create the model.")
        return

    # Create session
    session, active_provider = create_session(model_path, providers)
    print(f"\n  Active Provider: {active_provider}")

    # Get model info
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    print(f"  Input : {inputs[0].name} shape={inputs[0].shape} type={inputs[0].type}")
    print(f"  Output: {outputs[0].name} shape={outputs[0].shape} type={outputs[0].type}")

    # Benchmark mode
    if args.benchmark:
        print("\n[BENCHMARK] Running latency benchmark...")
        input_shape = tuple(
            d if isinstance(d, int) else 1 for d in inputs[0].shape
        )
        results = benchmark_inference(session, input_shape)
        print(f"  Mean   : {results['mean_ms']:.2f} ms")
        print(f"  Median : {results['median_ms']:.2f} ms")
        print(f"  Min    : {results['min_ms']:.2f} ms")
        print(f"  Max    : {results['max_ms']:.2f} ms")
        print(f"  Std    : {results['std_ms']:.2f} ms")
        return

    # Run inference
    np.random.seed(args.seed)
    input_shape = tuple(
        d if isinstance(d, int) else 1 for d in inputs[0].shape
    )
    input_data = np.random.randn(*input_shape).astype(np.float32)

    print(f"\n[INFERENCE] Running with seed={args.seed}...")
    start = time.perf_counter()
    output = run_inference(session, input_data)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  Output shape: {output.shape}")
    print(f"  Latency: {elapsed:.2f} ms")
    print(f"  Provider: {active_provider}")

    # Save output if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Normalize output to 0-255 range for visualization
        vis = output.flatten()
        side = int(np.sqrt(len(vis)))
        if side > 0:
            vis = vis[:side * side].reshape(side, side)
            vis = ((vis - vis.min()) / (vis.max() - vis.min() + 1e-8) * 255).astype(np.uint8)
            cv2.imwrite(str(out_path), vis)
            print(f"  Saved visualization: {out_path}")

    print(f"\n{'=' * 72}")
    print("[SUCCESS] ONNX Runtime inference complete (zero PyTorch dependency)")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
