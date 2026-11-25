# ===----------------------------------------------------------------------===
# The Buckyball compiler of one simulation, the shared library
# libbuckyball-midend.so: the device, hart, acc and frontend dialects, the bank
# interfaces of the acc dialect, the framework passes (lower-device,
# lower-hart, the bank optimizations, lower-acc, lower-frontend-fake-op), the
# dialect of every Ball of the device and the design.pb schema. It is the
# buddy-opt plugin entry of every level: the plugins of the devices, harts and
# threads (stack/compiler/plugin) hold only their design's passes and link it,
# so one buddy-opt loads the plugins of several levels together.
#
# Built by the stack top of each simulation entry, configured with
# DESIGN_CMAKE (generated from design.pb):
#   DESIGN_MLIR             MLIR CMake package of the LLVM that buddy-mlir uses
#   DESIGN_REPO             the repository: the design.pb schema and the Balls' messages
#   DESIGN_BALL_STACKS      stack/src directories of the Balls of the device
#
# The passes read the banks, Balls and ISA of their target from design.pb when
# they run (--design-pb, --design-target).
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(buckyball-midend LANGUAGES C CXX)
include("${DESIGN_CMAKE}")
set(BUCKYBALL_BALLS ${DESIGN_BALL_STACKS})

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
include_directories("${SOURCE}" "${SOURCE}/Dialect" "${SOURCE}/Dialect/Acc" "${CMAKE_CURRENT_BINARY_DIR}")

set(LLVM_TARGET_DEFINITIONS "${SOURCE}/Dialect/Acc/Acc.td")
mlir_tablegen(Acc.h.inc -gen-op-decls)
mlir_tablegen(Acc.cpp.inc -gen-op-defs)
mlir_tablegen(AccDialect.h.inc -gen-dialect-decls -dialect=bbacc)
mlir_tablegen(AccDialect.cpp.inc -gen-dialect-defs -dialect=bbacc)
add_public_tablegen_target(AccIncGen)

set(LLVM_TARGET_DEFINITIONS "${SOURCE}/Dialect/Device/Device.td")
mlir_tablegen(Device.h.inc -gen-op-decls)
mlir_tablegen(Device.cpp.inc -gen-op-defs)
mlir_tablegen(DeviceDialect.h.inc -gen-dialect-decls -dialect=device)
mlir_tablegen(DeviceDialect.cpp.inc -gen-dialect-defs -dialect=device)
add_public_tablegen_target(DeviceIncGen)

set(LLVM_TARGET_DEFINITIONS "${SOURCE}/Dialect/Hart/Hart.td")
mlir_tablegen(Hart.h.inc -gen-op-decls)
mlir_tablegen(Hart.cpp.inc -gen-op-defs)
mlir_tablegen(HartDialect.h.inc -gen-dialect-decls -dialect=hart)
mlir_tablegen(HartDialect.cpp.inc -gen-dialect-defs -dialect=hart)
add_public_tablegen_target(HartIncGen)

set(LLVM_TARGET_DEFINITIONS "${SOURCE}/Dialect/Frontend/Frontend.td")
mlir_tablegen(Frontend.h.inc -gen-op-decls)
mlir_tablegen(Frontend.cpp.inc -gen-op-defs)
mlir_tablegen(FrontendDialect.h.inc -gen-dialect-decls -dialect=frontend)
mlir_tablegen(FrontendDialect.cpp.inc -gen-dialect-defs -dialect=frontend)
add_public_tablegen_target(FrontendIncGen)

set(LLVM_TARGET_DEFINITIONS "${SOURCE}/Dialect/Acc/BankInterfaces.td")
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

add_library(buckyball-midend SHARED ${FRAMEWORK_SOURCES} ${BALL_SOURCES} ${PROTO_SOURCES})
add_dependencies(
  buckyball-midend
  AccIncGen
  DeviceIncGen
  HartIncGen
  FrontendIncGen
  BankInterfacesIncGen
  ${BALL_INC_GENS}
)
set_target_properties(buckyball-midend
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
# RTTI and exceptions as the LLVM build: off.
llvm_update_compile_flags(buckyball-midend)
# The MLIR symbols come from the buddy-opt that loads the plugins.
string(REPLACE "-Wl,-z,defs" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
target_link_libraries(buckyball-midend
  PUBLIC PkgConfig::PROTOBUF
)
