# ===----------------------------------------------------------------------===
# The engine part of a core's compiler, as the object library
# `engine-compiler`: lower-to-stream, which issues Ball instructions through
# the engine instead of the core. A core with hardware threads adds it to its
# own buddy-opt plugin.
#
# Inputs, set by the including core stack from its generated design.cmake:
#   DESIGN_MLIR   MLIR CMake package of the LLVM that buddy-mlir uses
# ===----------------------------------------------------------------------===
find_package(MLIR REQUIRED CONFIG PATHS "${DESIGN_MLIR}" NO_DEFAULT_PATH)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(HandleLLVMOptions)
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_library(engine-compiler OBJECT StreamLowering.cpp)
set_target_properties(engine-compiler
  PROPERTIES POSITION_INDEPENDENT_CODE ON
)
# RTTI and exceptions as the LLVM build: off.
llvm_update_compile_flags(engine-compiler)
