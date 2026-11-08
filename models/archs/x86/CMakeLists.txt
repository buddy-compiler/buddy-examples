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

# ===----------------------------------------------------------------------===
# The generic midend: buddy-opt loads libgeneric-compiler.so for the frontend
# dialect and lower-frontend-fake-op, which computes the frontend operations
# in software.
# ===----------------------------------------------------------------------===
include(ExternalProject)
set(GENERIC_COMPILER "${CMAKE_BINARY_DIR}/generic-compiler/libgeneric-compiler.so")
ExternalProject_Add(
  generic-compiler
  SOURCE_DIR "${STACK_ROOT}/compiler/generic"
  BINARY_DIR "${CMAKE_BINARY_DIR}/generic-compiler"
  CMAKE_ARGS "-DMLIR_DIR=${BUDDY_SOURCE}/llvm/build/lib/cmake/mlir"
  INSTALL_COMMAND ""
  BUILD_ALWAYS ON
  BUILD_BYPRODUCTS "${GENERIC_COMPILER}"
)

# ===----------------------------------------------------------------------===
# The x86 backend's own passes: buddy-opt loads libx86-compiler.so (the
# framework's frontend dialect and fuse-mxint8-weights).
# ===----------------------------------------------------------------------===
set(X86_COMPILER "${CMAKE_BINARY_DIR}/x86-compiler/libx86-compiler.so")
ExternalProject_Add(
  x86-compiler
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/compiler"
  BINARY_DIR "${CMAKE_BINARY_DIR}/x86-compiler"
  CMAKE_ARGS "-DMLIR_DIR=${BUDDY_SOURCE}/llvm/build/lib/cmake/mlir" "-DFRAMEWORK_MIDEND=${STACK_ROOT}/compiler/midend"
  INSTALL_COMMAND ""
  BUILD_ALWAYS ON
  BUILD_BYPRODUCTS "${X86_COMPILER}"
)

if(MODEL)
  add_subdirectory("${MODEL}")
else()
  add_subdirectory(alexnet)
  add_subdirectory(bert-base-uncased-emotion)
  add_subdirectory(deepseek-r1-distill-qwen-1-5b)
  add_subdirectory(gemma4-e2b-it)
  add_subdirectory(lenet)
  add_subdirectory(mobilenet-v3-small)
  add_subdirectory(pi05-base)
  add_subdirectory(resnet18)
  add_subdirectory(smolvla-base)
  add_subdirectory(stable-diffusion)
  add_subdirectory(whisper)
  add_subdirectory(yolo26n)
endif()
