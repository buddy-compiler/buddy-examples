# Use the buddy-mlir built by the main Buckyball repo, exposed via
# BUDDY_MLIR_BUILD_DIR (set by sourceme.sh in `nix develop`).
if(NOT DEFINED ENV{BUDDY_MLIR_BUILD_DIR})
  message(FATAL_ERROR
    "BUDDY_MLIR_BUILD_DIR is not set. Enter the nix develop shell so "
    "sourceme.sh is loaded, then re-run cmake.")
endif()
set(BUDDY_MLIR_DIR $ENV{BUDDY_MLIR_BUILD_DIR}/..)
set(BUDDY_BINARY_DIR $ENV{BUDDY_MLIR_BUILD_DIR}/bin)
set(RISCV_GNU_TOOLCHAIN $ENV{RISCV})

execute_process(
  COMMAND which python3
  OUTPUT_VARIABLE PYTHON3_ABS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
set(Python3_EXECUTABLE ${PYTHON3_ABS})
message(STATUS "Using Python3: ${Python3_EXECUTABLE}")

set(WORKLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MODEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/models)
set(ARCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/archs)
set(MODELTEST_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# ============================================================================
# Parse ARCH variable: ARCH="gemmini,other_arch,..."
# ============================================================================
if (DEFINED ARCH)
  string(REPLACE "," ";" ARCH_LIST ${ARCH})
  foreach(arch_item ${ARCH_LIST})
    string(STRIP ${arch_item} arch_item)
    string(TOUPPER ${arch_item} arch_upper)
    set(ARCH_${arch_upper} ON)
    message(STATUS "Enabled architecture: ${arch_item}")
  endforeach()
endif()

# ============================================================================
# Parse MODEL variable: MODEL="lenet,resnet18,bert,..."
# ============================================================================
if (DEFINED MODEL)
  string(REPLACE "," ";" MODEL_LIST ${MODEL})
  foreach(model_item ${MODEL_LIST})
    string(STRIP ${model_item} model_item)
    string(TOUPPER ${model_item} model_upper)
    set(MODEL_${model_upper} ON)
    message(STATUS "Enabled model: ${model_item}")
  endforeach()
endif()

add_subdirectory(lib)
add_subdirectory(models)
add_subdirectory(archs)
