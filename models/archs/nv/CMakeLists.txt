# ===----------------------------------------------------------------------===
# Detect GPU 0 compute capability and form the NVVM target (e.g. 7.5 -> sm_75).
# ===----------------------------------------------------------------------===
execute_process(
  COMMAND nvidia-smi --id=0 --query-gpu=compute_cap --format=csv,noheader
  OUTPUT_VARIABLE NV_CUBIN_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
string(REPLACE "." "" NV_CUBIN_VERSION "sm_${NV_CUBIN_VERSION}")
message(STATUS "NVIDIA GPU target: ${NV_CUBIN_VERSION}")

# ===----------------------------------------------------------------------===
# Image dependencies
# NVIDIA model executables use the host libpng provided by Nix via PNG::PNG.
# Keep this lookup in the NV backend: RISC-V executables require a libpng
# built for RISC-V and cannot link against the host library.
# ===----------------------------------------------------------------------===
find_package(PNG REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs HINTS "${TOOLCHAIN}")

# ===----------------------------------------------------------------------===
# Shared MLIR runtime utilities and tracing for NVIDIA models
# ===----------------------------------------------------------------------===
add_library(nv-runtime OBJECT EXCLUDE_FROM_ALL "${LIB_DIR}/CRunnerUtils.cpp" "${LIB_DIR}/native_trace.cpp"
  "${RUNTIME_DIR}/trace/trace.cpp"
)
target_include_directories(nv-runtime
  PRIVATE "${RUNTIME_DIR}/trace"
          "${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine"
)
set_target_properties(nv-runtime
  PROPERTIES POSITION_INDEPENDENT_CODE ON
)

if(MODEL)
  add_subdirectory("${MODEL}")
else()
  add_subdirectory(alexnet)
  add_subdirectory(bert-base-uncased-emotion)
  add_subdirectory(deepseek-r1-distill-qwen-1.5b)
  add_subdirectory(gemma4-e2b-it)
  add_subdirectory(lenet)
  add_subdirectory(minimax-h3-fl2va)
  add_subdirectory(minimax-h3-ref2va)
  add_subdirectory(mobilenet-v3-small)
  add_subdirectory(qwen3-8b)
  add_subdirectory(resnet18)
  add_subdirectory(stable-diffusion)
  add_subdirectory(whisper)
  add_subdirectory(yolo26n)
endif()
