# ===----------------------------------------------------------------------===
# The passes of a design at one level (convert-linalg-to-device, -hart or
# -acc), as the object library `buckyball-compiler` that the stack of that
# level links into its buddy-opt plugin. The framework, the Balls and the
# design.pb schema are libbuckyball-midend.so (stack/compiler/midend), built
# once per simulation entry and linked by every plugin.
#
# Inputs, set by the including stack from its generated design.cmake:
#   DESIGN_MLIR             MLIR CMake package of the LLVM that buddy-mlir uses
#   DESIGN_MIDEND_BUILD     the build of libbuckyball-midend.so and its headers
#   DESIGN_BALL_STACKS      stack/src directories of the Balls of the device
#   BUCKYBALL_SOURCES       the design's pass sources of the including stack
# ===----------------------------------------------------------------------===
find_package(MLIR REQUIRED CONFIG PATHS "${DESIGN_MLIR}" NO_DEFAULT_PATH)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(HandleLLVMOptions)
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

set(MIDEND "${DESIGN_REPO}/stack/compiler/midend")
include_directories(
  "${MIDEND}"
  "${MIDEND}/Dialect"
  "${MIDEND}/Dialect/Acc"
  "${DESIGN_MIDEND_BUILD}"
  ${DESIGN_BALL_STACKS}
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(PROTOBUF REQUIRED IMPORTED_TARGET protobuf)
add_library(buckyball-compiler OBJECT ${BUCKYBALL_SOURCES})
set_target_properties(buckyball-compiler
  PROPERTIES POSITION_INDEPENDENT_CODE ON
)
# RTTI and exceptions as the LLVM build: off.
llvm_update_compile_flags(buckyball-compiler)
target_link_libraries(buckyball-compiler
  PUBLIC "${DESIGN_MIDEND_BUILD}/libbuckyball-midend.so"
         PkgConfig::PROTOBUF
)
