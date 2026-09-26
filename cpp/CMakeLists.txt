cmake_minimum_required(VERSION 3.20)
project(sultan_hassan_onnx_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Find ONNX Runtime ─────────────────────────────────────────────────────
# Set ONNXRUNTIME_DIR to the root of your ONNX Runtime installation
# e.g., cmake -DONNXRUNTIME_DIR="C:/onnxruntime-win-x64-gpu-1.18.0" ..
if(NOT DEFINED ONNXRUNTIME_DIR)
    message(STATUS "ONNXRUNTIME_DIR not set. Trying system paths...")
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        PATHS /usr/local/include/onnxruntime
              /usr/include/onnxruntime
              "C:/onnxruntime/include"
    )
    find_library(ONNXRUNTIME_LIB onnxruntime
        PATHS /usr/local/lib
              /usr/lib
              "C:/onnxruntime/lib"
    )
else()
    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
    find_library(ONNXRUNTIME_LIB onnxruntime PATHS "${ONNXRUNTIME_DIR}/lib")
endif()

if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime not found. Set -DONNXRUNTIME_DIR=<path>")
endif()

message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime lib: ${ONNXRUNTIME_LIB}")

# ── DirectML support (Windows only) ───────────────────────────────────────
option(USE_DML "Enable DirectML execution provider" OFF)

# ── Build inference binary ────────────────────────────────────────────────
add_executable(infer_onnx infer_onnx.cpp)
target_include_directories(infer_onnx PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(infer_onnx PRIVATE ${ONNXRUNTIME_LIB})

if(USE_DML)
    target_compile_definitions(infer_onnx PRIVATE USE_DML)
endif()

# ── Install ───────────────────────────────────────────────────────────────
install(TARGETS infer_onnx RUNTIME DESTINATION bin)
