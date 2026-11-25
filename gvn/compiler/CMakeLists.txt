# ===----------------------------------------------------------------------===
# The GVN stimulus compiler, as the buddy-opt dialect and pass plugin
# libgvn-compiler.so: the gvn dialect (gvn.check) and the passes that cut a
# compiled graph into slices (NPU side) and expose the values at the slice
# boundaries (GPU side).
#
# Input: MLIR_DIR, the MLIR CMake package of the LLVM that buddy-mlir uses.
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.22)
project(gvn-compiler LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}")

set(LLVM_TARGET_DEFINITIONS "${CMAKE_CURRENT_SOURCE_DIR}/Gvn.td")
mlir_tablegen(Gvn.h.inc -gen-op-decls)
mlir_tablegen(Gvn.cpp.inc -gen-op-defs)
mlir_tablegen(GvnDialect.h.inc -gen-dialect-decls -dialect=gvn)
mlir_tablegen(GvnDialect.cpp.inc -gen-dialect-defs -dialect=gvn)
add_public_tablegen_target(GvnIncGen)

add_library(gvn-compiler MODULE "${CMAKE_CURRENT_SOURCE_DIR}/Plugin.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/GvnDialect.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/InsertChecks.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/Slice.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/SliceOp.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/Expose.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/ForwardResults.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/Import.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/GpuProgram.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/HostShapes.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/ManagedMemory.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/PackedEntry.cpp"
)
add_dependencies(gvn-compiler GvnIncGen)
llvm_update_compile_flags(gvn-compiler)
set_target_properties(gvn-compiler
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
