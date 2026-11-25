# ===----------------------------------------------------------------------===
# GVN, the stimulus path that cuts a model into slices: its buddy-opt plugins.
#   gvn-compiler      libgvn-compiler.so: the gvn dialect and the passes that
#                     slice the NPU graph and expose the values for the GPU
#   generic-compiler  libgeneric-compiler.so: the generic midend of the GPU side
#   nv-compiler       libnv-compiler.so: the nv passes of the GPU side
# Not part of the model compile path: no recipe loads these.
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.22)
project(gvn NONE)
include(ExternalProject)
get_filename_component(STACK_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(MLIR_DIR "${STACK_ROOT}/thirdparty/buddy-mlir/llvm/build/lib/cmake/mlir")

ExternalProject_Add(
  gvn-compiler
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/compiler"
  BINARY_DIR "${CMAKE_BINARY_DIR}/gvn-compiler"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DMLIR_DIR=${MLIR_DIR}" -DCMAKE_BUILD_TYPE=Release
  BUILD_ALWAYS ON
  INSTALL_COMMAND ""
)

ExternalProject_Add(
  generic-compiler
  SOURCE_DIR "${STACK_ROOT}/compiler/generic"
  BINARY_DIR "${CMAKE_BINARY_DIR}/generic-compiler"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DMLIR_DIR=${MLIR_DIR}" -DCMAKE_BUILD_TYPE=Release
  BUILD_ALWAYS ON
  INSTALL_COMMAND ""
)

ExternalProject_Add(
  nv-compiler
  SOURCE_DIR "${STACK_ROOT}/models/archs/nv/compiler"
  BINARY_DIR "${CMAKE_BINARY_DIR}/nv-compiler"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DMLIR_DIR=${MLIR_DIR}" -DCMAKE_BUILD_TYPE=Release "-DFRAMEWORK_MIDEND=${STACK_ROOT}/compiler/midend"
  BUILD_ALWAYS ON
  INSTALL_COMMAND ""
)
