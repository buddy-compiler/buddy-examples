# ===----------------------------------------------------------------------===
# The NVIDIA backend's own compiler passes, as the buddy-opt dialect and pass
# plugin libnv-compiler.so: the framework's frontend dialect and
# fuse-mxint8-weights, which reads packed MXINT8 weights inside the
# contractions using them, and linalg-to-gpu-kernels, which maps every
# linalg op on buffers to a thread-block kernel.
#
# Inputs:
#   MLIR_DIR          MLIR CMake package of the LLVM that buddy-mlir uses
#   FRAMEWORK_MIDEND  stack/compiler/midend: the frontend dialect
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.22)
project(nv-compiler LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

# Built with the flags of the LLVM that buddy-opt links, so the plugin matches it.
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
include_directories("${FRAMEWORK_MIDEND}/Dialect" "${CMAKE_CURRENT_BINARY_DIR}")

set(LLVM_TARGET_DEFINITIONS "${FRAMEWORK_MIDEND}/Dialect/Frontend/Frontend.td")
mlir_tablegen(Frontend.h.inc -gen-op-decls)
mlir_tablegen(Frontend.cpp.inc -gen-op-defs)
mlir_tablegen(FrontendDialect.h.inc -gen-dialect-decls -dialect=frontend)
mlir_tablegen(FrontendDialect.cpp.inc -gen-dialect-defs -dialect=frontend)
add_public_tablegen_target(FrontendIncGen)

add_library(nv-compiler MODULE "${CMAKE_CURRENT_SOURCE_DIR}/Plugin.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/FuseMXINT8Weights.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/LinalgToGPUKernels.cpp"
  "${FRAMEWORK_MIDEND}/Dialect/Frontend/IR/FrontendDialect.cpp"
)
add_dependencies(nv-compiler FrontendIncGen)
# RTTI and exceptions as the LLVM build: off.
llvm_update_compile_flags(nv-compiler)
set_target_properties(nv-compiler
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
