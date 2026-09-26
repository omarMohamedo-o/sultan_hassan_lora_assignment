/*
 * Sultan Hassan FLUX LoRA – C++ ONNX Runtime Inference
 * ====================================================
 * Maximum performance inference using the ONNX C++ API.
 * Zero Python dependency. Supports DirectML, CUDA, TensorRT, CPU.
 *
 * Build (Windows with MSVC):
 *   cl /EHsc /O2 /I "path/to/onnxruntime/include" infer_onnx.cpp ^
 *      /link /LIBPATH:"path/to/onnxruntime/lib" onnxruntime.lib
 *
 * Build (Linux with GCC):
 *   g++ -O2 -o infer_onnx infer_onnx.cpp \
 *       -I/usr/local/include/onnxruntime \
 *       -L/usr/local/lib -lonnxruntime
 *
 * Usage:
 *   ./infer_onnx models/onnx/sltnhsn_flux_lora.onnx [provider]
 *   ./infer_onnx models/onnx/sltnhsn_flux_lora.onnx DML
 *   ./infer_onnx models/onnx/sltnhsn_flux_lora.onnx CUDA
 *   ./infer_onnx models/onnx/sltnhsn_flux_lora.onnx CPU
 */

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

// ── Helpers ──────────────────────────────────────────────────────────────────

static void print_separator() {
    printf("========================================================================\n");
}

static double elapsed_ms(
    std::chrono::high_resolution_clock::time_point start,
    std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <model.onnx> [provider: CPU|CUDA|DML|TRT]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    std::string provider = (argc >= 3) ? argv[2] : "CPU";

    print_separator();
    printf("SULTAN HASSAN – C++ ONNX Runtime Inference\n");
    print_separator();
    printf("  Model    : %s\n", model_path);
    printf("  Provider : %s\n", provider.c_str());

    try {
        // ── Environment & Session Options ────────────────────────────────
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SultanHassanLoRA");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(0);  // Use all cores
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // ── Execution Provider Selection ─────────────────────────────────
        if (provider == "DML") {
#ifdef USE_DML
            Ort::ThrowOnError(
                OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0));
            printf("  [OK] DirectML provider enabled (GPU device 0)\n");
#else
            printf("  [WARN] DirectML not compiled in, falling back to CPU\n");
#endif
        } else if (provider == "CUDA") {
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = 0;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            printf("  [OK] CUDA provider enabled (GPU device 0)\n");
        } else if (provider == "TRT") {
            OrtTensorRTProviderOptions trt_options{};
            trt_options.device_id = 0;
            trt_options.trt_fp16_enable = 1;
            session_options.AppendExecutionProvider_TensorRT(trt_options);
            printf("  [OK] TensorRT provider enabled (FP16, GPU device 0)\n");
        } else {
            printf("  [OK] CPU provider (default)\n");
        }

        // ── Load Model ───────────────────────────────────────────────────
#ifdef _WIN32
        // Windows: convert to wide string
        int len = MultiByteToWideChar(CP_UTF8, 0, model_path, -1, nullptr, 0);
        std::vector<wchar_t> wpath(len);
        MultiByteToWideChar(CP_UTF8, 0, model_path, -1, wpath.data(), len);
        Ort::Session session(env, wpath.data(), session_options);
#else
        Ort::Session session(env, model_path, session_options);
#endif

        Ort::AllocatorWithDefaultOptions allocator;

        // ── Input Info ───────────────────────────────────────────────────
        size_t num_inputs = session.GetInputCount();
        size_t num_outputs = session.GetOutputCount();
        printf("\n  Inputs : %zu\n", num_inputs);
        printf("  Outputs: %zu\n", num_outputs);

        auto input_name = session.GetInputNameAllocated(0, allocator);
        auto output_name = session.GetOutputNameAllocated(0, allocator);
        auto input_info = session.GetInputTypeInfo(0);
        auto tensor_info = input_info.GetTensorTypeAndShapeInfo();
        auto input_shape = tensor_info.GetShape();

        printf("  Input  name=%s shape=[", input_name.get());
        for (size_t i = 0; i < input_shape.size(); i++) {
            printf("%lld%s", input_shape[i], i < input_shape.size() - 1 ? "," : "");
        }
        printf("]\n");

        // Replace dynamic dims with 1
        for (auto& d : input_shape) {
            if (d < 0) d = 1;
        }

        // ── Prepare Input ────────────────────────────────────────────────
        int64_t total_elements = 1;
        for (auto d : input_shape) total_elements *= d;

        std::vector<float> input_data(total_elements);
        std::srand(101);
        for (auto& v : input_data) {
            v = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
        }

        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size());

        // ── Warmup ───────────────────────────────────────────────────────
        printf("\n  Warming up (3 runs)...\n");
        const char* input_names[] = {input_name.get()};
        const char* output_names[] = {output_name.get()};

        for (int i = 0; i < 3; i++) {
            session.Run(Ort::RunOptions{nullptr},
                        input_names, &input_tensor, 1,
                        output_names, 1);
        }

        // ── Benchmark ────────────────────────────────────────────────────
        const int NUM_RUNS = 20;
        std::vector<double> latencies(NUM_RUNS);
        printf("  Benchmarking (%d runs)...\n", NUM_RUNS);

        for (int i = 0; i < NUM_RUNS; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto output_tensors = session.Run(
                Ort::RunOptions{nullptr},
                input_names, &input_tensor, 1,
                output_names, 1);
            auto t1 = std::chrono::high_resolution_clock::now();
            latencies[i] = elapsed_ms(t0, t1);
        }

        std::sort(latencies.begin(), latencies.end());
        double mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) / NUM_RUNS;
        double median = latencies[NUM_RUNS / 2];
        double p99 = latencies[static_cast<int>(NUM_RUNS * 0.99)];

        // ── Results ──────────────────────────────────────────────────────
        printf("\n");
        print_separator();
        printf("  BENCHMARK RESULTS (%s)\n", provider.c_str());
        print_separator();
        printf("  Mean   : %.3f ms\n", mean);
        printf("  Median : %.3f ms\n", median);
        printf("  Min    : %.3f ms\n", latencies.front());
        printf("  Max    : %.3f ms\n", latencies.back());
        printf("  P99    : %.3f ms\n", p99);
        print_separator();
        printf("  [SUCCESS] C++ ONNX inference complete (zero Python dependency)\n");
        print_separator();

    } catch (const Ort::Exception& e) {
        printf("[ERROR] ONNX Runtime: %s\n", e.what());
        return 1;
    }

    return 0;
}
