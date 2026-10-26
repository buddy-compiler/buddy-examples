# ===----------------------------------------------------------------------===
# The Buckyball compiler of one compiler target, as the object library
# `buckyball-compiler`: the custom and bank dialects, the framework passes and the
# lower-buckyball part of every Ball. The thread or core stack that owns the
# target adds form-custom-ops and convert-linalg-to-buckyball to its plugin.
#
# Inputs, set by the including stack from its generated design.cmake:
#   DESIGN_MLIR             MLIR CMake package of the LLVM that buddy-mlir uses
#   BUCKYBALL_BALLS         stack/src directories of the target's Balls
#   BUCKYBALL_TARGET_TABLE  target.inc: banks, Balls and ISA of the target
#   BUCKYBALL_SOURCES       compiler sources of the including stack itself
# ===----------------------------------------------------------------------===
# Built with the flags of the LLVM that buddy-opt links, so the plugin matches it.
find_package(MLIR REQUIRED CONFIG PATHS "${DESIGN_MLIR}" NO_DEFAULT_PATH)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

set(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}")
configure_file("${BUCKYBALL_TARGET_TABLE}" "${CMAKE_CURRENT_BINARY_DIR}/BuckyballTargetRegistry.inc" COPYONLY)
include_directories("${SOURCE}" "${SOURCE}/Dialect" "${SOURCE}/Dialect/Buckyball" "${CMAKE_CURRENT_BINARY_DIR}")

set(LLVM_TARGET_DEFINITIONS "${SOURCE}/Dialect/Buckyball/Buckyball.td")
mlir_tablegen(Buckyball.h.inc -gen-op-decls)
mlir_tablegen(Buckyball.cpp.inc -gen-op-defs)
mlir_tablegen(BuckyballDialect.h.inc -gen-dialect-decls -dialect=buckyball)
mlir_tablegen(BuckyballDialect.cpp.inc -gen-dialect-defs -dialect=buckyball)
add_public_tablegen_target(BuckyballIncGen)

set(LLVM_TARGET_DEFINITIONS "${SOURCE}/Dialect/Custom/Custom.td")
mlir_tablegen(Custom.h.inc -gen-op-decls)
mlir_tablegen(Custom.cpp.inc -gen-op-defs)
mlir_tablegen(CustomDialect.h.inc -gen-dialect-decls -dialect=custom)
mlir_tablegen(CustomDialect.cpp.inc -gen-dialect-defs -dialect=custom)
add_public_tablegen_target(CustomIncGen)

file(GLOB_RECURSE FRAMEWORK_SOURCES CONFIGURE_DEPENDS "${SOURCE}/*.cpp")
set(BALL_SOURCES)
foreach(BALL IN LISTS BUCKYBALL_BALLS)
  file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "${BALL}/*.cpp")
  list(APPEND BALL_SOURCES ${SOURCES})
endforeach()
add_library(buckyball-compiler OBJECT ${FRAMEWORK_SOURCES} ${BALL_SOURCES} ${BUCKYBALL_SOURCES})
add_dependencies(buckyball-compiler BuckyballIncGen CustomIncGen)
set_target_properties(buckyball-compiler
  PROPERTIES POSITION_INDEPENDENT_CODE ON
)
# RTTI and exceptions as the LLVM build: off.
llvm_update_compile_flags(buckyball-compiler)
