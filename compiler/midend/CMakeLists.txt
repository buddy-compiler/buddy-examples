# ===----------------------------------------------------------------------===
# The Buckyball compiler of one compiler target, as the object library
# `buckyball-compiler`: the custom and bank dialects, the bank interfaces, the
# framework passes and the dialect of every Ball. The thread or core stack that owns the
# target adds form-custom-ops and convert-linalg-to-buckyball to its plugin.
#
# Inputs, set by the including stack from its generated design.cmake:
#   DESIGN_MLIR             MLIR CMake package of the LLVM that buddy-mlir uses
#   DESIGN_REPO             the repository: the design.pb schema and the Balls' messages
#   BUCKYBALL_BALLS         stack/src directories of the target's Balls
#   BUCKYBALL_SOURCES       compiler sources of the including stack itself
#
# The plugin reads the banks, Balls and ISA of its target from design.pb when
# it runs (--buckyball-design-pb, --buckyball-target), so it is built with the
# design.pb schema and the parameter messages of its Balls.
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

set(LLVM_TARGET_DEFINITIONS "${SOURCE}/Interfaces/BankInterfaces.td")
mlir_tablegen(BankInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(BankInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(BankInterfacesIncGen)

file(GLOB_RECURSE FRAMEWORK_SOURCES CONFIGURE_DEPENDS "${SOURCE}/*.cpp")
# Every Ball has its dialect in <ball>Ops.td, the operations it implements.
set(BALL_SOURCES)
set(BALL_INC_GENS)
foreach(BALL IN LISTS BUCKYBALL_BALLS)
  get_filename_component(BALL_NAME "${BALL}/../.." ABSOLUTE)
  get_filename_component(BALL_NAME "${BALL_NAME}" NAME)
  include_directories("${BALL}")
  set(LLVM_TARGET_DEFINITIONS "${BALL}/${BALL_NAME}Ops.td")
  mlir_tablegen(${BALL_NAME}Ops.h.inc -gen-op-decls)
  mlir_tablegen(${BALL_NAME}Ops.cpp.inc -gen-op-defs)
  mlir_tablegen(${BALL_NAME}Dialect.h.inc -gen-dialect-decls)
  mlir_tablegen(${BALL_NAME}Dialect.cpp.inc -gen-dialect-defs)
  add_public_tablegen_target(${BALL_NAME}IncGen)
  list(APPEND BALL_INC_GENS ${BALL_NAME}IncGen)
  file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "${BALL}/*.cpp")
  list(APPEND BALL_SOURCES ${SOURCES})
endforeach()

# The design.pb schema and the parameter messages of the target's Balls, for
# the nix protobuf (pkg-config) the plugin links.
find_package(PkgConfig REQUIRED)
pkg_check_modules(PROTOBUF REQUIRED IMPORTED_TARGET protobuf)
find_program(PROTOC protoc REQUIRED)
set(PROTO_DIR "${DESIGN_REPO}/bbdev/api/steps/config/scripts/proto")
add_custom_command(
  OUTPUT design.pb.cc
         design.pb.h
         hardware.pb.cc
         hardware.pb.h
  COMMAND "${PROTOC}" "--cpp_out=${CMAKE_CURRENT_BINARY_DIR}" "-I${PROTO_DIR}" "${PROTO_DIR}/design.proto" "${PROTO_DIR}/hardware.proto"
  DEPENDS "${PROTO_DIR}/design.proto"
          "${PROTO_DIR}/hardware.proto"
  VERBATIM
)
set(PROTO_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/design.pb.cc" "${CMAKE_CURRENT_BINARY_DIR}/hardware.pb.cc")
foreach(BALL IN LISTS BUCKYBALL_BALLS)
  get_filename_component(BALL_DIR "${BALL}/../.." ABSOLUTE)
  get_filename_component(BALL_NAME "${BALL_DIR}" NAME)
  add_custom_command(
    OUTPUT ${BALL_NAME}_params.pb.cc
           ${BALL_NAME}_params.pb.h
    COMMAND "${PROTOC}" "--cpp_out=${CMAKE_CURRENT_BINARY_DIR}" "-I${BALL_DIR}/configs" "${BALL_DIR}/configs/${BALL_NAME}_params.proto"
    DEPENDS "${BALL_DIR}/configs/${BALL_NAME}_params.proto"
    VERBATIM
  )
  list(APPEND PROTO_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/${BALL_NAME}_params.pb.cc")
endforeach()

add_library(buckyball-compiler OBJECT ${FRAMEWORK_SOURCES} ${BALL_SOURCES} ${BUCKYBALL_SOURCES} ${PROTO_SOURCES})
add_dependencies(buckyball-compiler BuckyballIncGen CustomIncGen BankInterfacesIncGen ${BALL_INC_GENS})
set_target_properties(buckyball-compiler
  PROPERTIES POSITION_INDEPENDENT_CODE ON
)
# RTTI and exceptions as the LLVM build: off.
llvm_update_compile_flags(buckyball-compiler)
target_link_libraries(buckyball-compiler
  PUBLIC PkgConfig::PROTOBUF
)
