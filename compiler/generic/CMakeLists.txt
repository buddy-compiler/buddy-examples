# ===----------------------------------------------------------------------===
# The generic midend of the targets without Balls (x86, nv), as the buddy-opt
# dialect and pass plugin libgeneric-compiler.so: the frontend dialect, and
# lower-frontend-fake-op, which computes every frontend operation in software.
#
# Input: MLIR_DIR, the MLIR CMake package of the LLVM that buddy-mlir uses.
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.22)
project(generic-compiler LANGUAGES C CXX)
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

set(MIDEND "${CMAKE_CURRENT_SOURCE_DIR}/../midend")
include_directories("${MIDEND}/Dialect" "${CMAKE_CURRENT_BINARY_DIR}")

set(LLVM_TARGET_DEFINITIONS "${MIDEND}/Dialect/Frontend/Frontend.td")
mlir_tablegen(Frontend.h.inc -gen-op-decls)
mlir_tablegen(Frontend.cpp.inc -gen-op-defs)
mlir_tablegen(FrontendDialect.h.inc -gen-dialect-decls -dialect=frontend)
mlir_tablegen(FrontendDialect.cpp.inc -gen-dialect-defs -dialect=frontend)
add_public_tablegen_target(FrontendIncGen)

add_library(generic-compiler MODULE "${CMAKE_CURRENT_SOURCE_DIR}/Plugin.cpp"
  "${MIDEND}/Dialect/Frontend/IR/FrontendDialect.cpp" "${MIDEND}/Conversion/LowerFrontendFakeOp.cpp"
)
add_dependencies(generic-compiler FrontendIncGen)
# RTTI and exceptions as the LLVM build: off.
llvm_update_compile_flags(generic-compiler)
set_target_properties(generic-compiler
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
