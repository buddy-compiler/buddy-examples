# ===----------------------------------------------------------------------===
# Host image libraries, MLIR runtime utilities and tracing for x86 models.
# ===----------------------------------------------------------------------===
find_package(PNG REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs HINTS "${TOOLCHAIN}")

# ===----------------------------------------------------------------------===
# Shared MLIR runtime utilities and tracing for x86 models
# ===----------------------------------------------------------------------===
add_library(x86-runtime OBJECT EXCLUDE_FROM_ALL "${LIB_DIR}/CRunnerUtils.cpp" "${LIB_DIR}/native_trace.cpp"
  "${RUNTIME_DIR}/trace/trace.cpp"
)
target_include_directories(x86-runtime
  PRIVATE "${RUNTIME_DIR}/trace"
          "${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine"
)
set_target_properties(x86-runtime
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
