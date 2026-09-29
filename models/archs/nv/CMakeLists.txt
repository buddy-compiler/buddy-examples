set(NV_DIR ${ARCH_DIR}/nv)

set(NV_LLVM_BUILD_DIR "${BUDDY_MLIR_DIR}/llvm/build" CACHE PATH "LLVM build directory")
set(NV_LLVM_TOOLS_BINARY_DIR ${NV_LLVM_BUILD_DIR}/bin)
set(NV_LLVM_LIBRARY_DIR ${NV_LLVM_BUILD_DIR}/lib)

if(NOT EXISTS ${NV_LLVM_TOOLS_BINARY_DIR}/mlir-opt)
  message(FATAL_ERROR "mlir-opt not found at ${NV_LLVM_TOOLS_BINARY_DIR}")
endif()
if(NOT EXISTS ${BUDDY_BINARY_DIR}/buddy-opt)
  message(FATAL_ERROR "buddy-opt not found at ${BUDDY_BINARY_DIR}")
endif()

find_library(NV_MLIR_CUDA_RUNTIME_LIB
  NAMES mlir_cuda_runtime
  PATHS ${NV_LLVM_LIBRARY_DIR}
  NO_DEFAULT_PATH
)
if(NOT NV_MLIR_CUDA_RUNTIME_LIB)
  message(FATAL_ERROR
    "libmlir_cuda_runtime not found under ${NV_LLVM_LIBRARY_DIR}. "
    "Rebuild LLVM/MLIR with -DLLVM_TARGETS_TO_BUILD=host;NVPTX "
    "-DMLIR_ENABLE_CUDA_RUNNER=ON.")
endif()

find_library(NV_MLIR_C_RUNNER_UTILS_LIB NAMES mlir_c_runner_utils PATHS ${NV_LLVM_LIBRARY_DIR} NO_DEFAULT_PATH)
find_library(NV_MLIR_RUNNER_UTILS_LIB NAMES mlir_runner_utils PATHS ${NV_LLVM_LIBRARY_DIR} NO_DEFAULT_PATH)
find_library(NV_MLIR_ASYNC_RUNTIME_LIB NAMES mlir_async_runtime PATHS ${NV_LLVM_LIBRARY_DIR} NO_DEFAULT_PATH)
if(NOT NV_MLIR_C_RUNNER_UTILS_LIB OR NOT NV_MLIR_RUNNER_UTILS_LIB)
  message(FATAL_ERROR "mlir runner utils not found under ${NV_LLVM_LIBRARY_DIR}")
endif()

set(BUDDY_BUILD_LIB_DIR ${BUDDY_MLIR_DIR}/build/lib)
find_library(NV_BUDDY_LIB_DIP NAMES BuddyLibDIP PATHS ${BUDDY_BUILD_LIB_DIR} NO_DEFAULT_PATH)
find_library(NV_BUDDY_LIB_DAP NAMES BuddyLibDAP PATHS ${BUDDY_BUILD_LIB_DIR} NO_DEFAULT_PATH)

find_package(PNG REQUIRED)

set(NV_CUBIN_CHIP sm_80 CACHE STRING "CUDA target architecture")
set(NV_LOWER_TO_NVVM_OPTION
  "cubin-chip=${NV_CUBIN_CHIP} cubin-features=+ptx71 cubin-format=fatbin")
set(NV_CONVERT_MEMCPY_TO_GPU_OPTION_DISABLE_PROCESS_ARG "process-args=0")

add_library(NvHostCRunnerUtils OBJECT ${MODELTEST_LIB_DIR}/CRunnerUtils.cpp)
target_include_directories(NvHostCRunnerUtils PUBLIC
  ${BUDDY_MLIR_DIR}/llvm/mlir/include/mlir/ExecutionEngine)
set_target_properties(NvHostCRunnerUtils PROPERTIES
  CXX_STANDARD 17
  POSITION_INDEPENDENT_CODE ON)

set(NV_INTERFACE_DIR ${BUDDY_MLIR_DIR}/frontend/Interfaces)

message(STATUS "NV arch: LLVM tools ${NV_LLVM_TOOLS_BINARY_DIR}")
message(STATUS "NV arch: cubin-chip=${NV_CUBIN_CHIP}")
message(STATUS "NV arch: cuda runtime ${NV_MLIR_CUDA_RUNTIME_LIB}")
if(NV_BUDDY_LIB_DIP)
  message(STATUS "NV arch: BuddyLibDIP ${NV_BUDDY_LIB_DIP}")
endif()
if(NV_BUDDY_LIB_DAP)
  message(STATUS "NV arch: BuddyLibDAP ${NV_BUDDY_LIB_DAP}")
endif()

if((MODEL_RESNET18 OR MODEL_MOBILENETV3 OR MODEL_YOLO26 OR MODEL_BERT OR MODEL_STABLEDIFFUSION)
   AND NOT NV_BUDDY_LIB_DIP)
  message(FATAL_ERROR "BuddyLibDIP not found under ${BUDDY_BUILD_LIB_DIR} (required by selected NV models)")
endif()
if(MODEL_WHISPER AND NOT NV_BUDDY_LIB_DAP)
  message(FATAL_ERROR "BuddyLibDAP not found under ${BUDDY_BUILD_LIB_DIR} (required by NV Whisper)")
endif()

if (MODEL_LENET)
  add_subdirectory(lenet)
endif()
if (MODEL_RESNET18)
  add_subdirectory(resnet18)
endif()
if (MODEL_MOBILENETV3)
  add_subdirectory(mobilenet-v3-small)
endif()
if (MODEL_YOLO26)
  add_subdirectory(yolo26n)
endif()
if (MODEL_BERT)
  add_subdirectory(bert-base-uncased-emotion)
endif()
if (MODEL_DEEPSEEKR1)
  add_subdirectory(deepseek-r1-distill-qwen-1.5b)
endif()
if (MODEL_QWEN3)
  add_subdirectory(qwen3-8b)
endif()
if (MODEL_GEMMA4)
  add_subdirectory(gemma4-e2b-it)
endif()
if (MODEL_STABLEDIFFUSION)
  add_subdirectory(stable-diffusion)
endif()
if (MODEL_WHISPER)
  add_subdirectory(whisper)
endif()
if (MODEL_MINIMAX_H3_FL2VA)
  add_subdirectory(minimax-h3-fl2va)
endif()
if (MODEL_MINIMAX_H3_REF2VA)
  add_subdirectory(minimax-h3-ref2va)
endif()
