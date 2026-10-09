cmake_minimum_required(VERSION 3.22)
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(SERVING_SOURCE "${CMAKE_CURRENT_LIST_DIR}")
if(NOT DEFINED CHIP OR NOT DEFINED MODEL)
  message(FATAL_ERROR "Set CHIP and MODEL")
endif()
set(MODEL_RECIPE "${REPO_ROOT}/examples/models/${CHIP}/${MODEL}" CACHE PATH "Model design directory")
if(NOT EXISTS "${MODEL_RECIPE}/CMakeLists.txt")
  message(FATAL_ERROR "Chip ${CHIP} does not provide a ${MODEL} model recipe")
endif()
set(TOOLCHAIN "${REPO_ROOT}/result" CACHE PATH "RISC-V toolchain")
set(CMAKE_C_COMPILER "${TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "${TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++")
project(Serving C CXX)
include(ExternalProject)
ExternalProject_Add(model-loader
  SOURCE_DIR "${SERVING_SOURCE}/loader"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/loader"
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
  BUILD_ALWAYS TRUE
  INSTALL_COMMAND "")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(COMPILER_BUILD "${REPO_ROOT}/stack/compiler/thirdparty/buddy-mlir/build/${CHIP}" CACHE PATH "Compiler build")
set(BUDDY_SOURCE "${REPO_ROOT}/stack/compiler/thirdparty/buddy-mlir")
set(PYTHON "${TOOLCHAIN}/bin/python3")
set(CHIP_PB "${REPO_ROOT}/examples/chips/${CHIP}/configs/generated/chip.pb" CACHE FILEPATH "Chip configuration")
set(ISA_DIR "${CMAKE_CURRENT_BINARY_DIR}/isa")
execute_process(COMMAND "${PYTHON}" "${REPO_ROOT}/stack/compiler/scripts/pb_to_target_registry.py"
  --repo "${REPO_ROOT}" --chip-pb "${CHIP_PB}" --isa-dir "${ISA_DIR}" --print-targets
  OUTPUT_VARIABLE TARGETS OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE "\n" ";" TARGETS "${TARGETS}")
execute_process(COMMAND "${PYTHON}" "${REPO_ROOT}/stack/compiler/scripts/pb_to_target_registry.py"
  --repo "${REPO_ROOT}" --chip-pb "${CHIP_PB}" --print-runtime-targets
  OUTPUT_VARIABLE RUNTIME_TARGETS OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE "\n" ";" RUNTIME_TARGETS "${RUNTIME_TARGETS}")
list(GET RUNTIME_TARGETS 0 PARAMS_TARGET)
file(STRINGS "${ISA_DIR}/${PARAMS_TARGET}/params.h" BANK_LINES_DECL REGEX "^#define BANK_LINES ")
string(REGEX REPLACE "^#define BANK_LINES ([0-9]+)$" "\\1" RUNTIME_BANK_LINES "${BANK_LINES_DECL}")
file(STRINGS "${ISA_DIR}/${PARAMS_TARGET}/params.h" MEMORY_PARAMS REGEX "^#define (BANK_WIDTH|VIRTUAL_BANK_NUM) ")
foreach(PROFILE IN LISTS RUNTIME_TARGETS)
  file(STRINGS "${ISA_DIR}/${PROFILE}/params.h" PARAMS REGEX "^#define (BANK_WIDTH|VIRTUAL_BANK_NUM) ")
  if(NOT PARAMS STREQUAL MEMORY_PARAMS)
    message(FATAL_ERROR "DMA runtime requires matching bank width and virtual bank count")
  endif()
  file(STRINGS "${ISA_DIR}/${PROFILE}/params.h" BANK_LINES_DECL REGEX "^#define BANK_LINES ")
  string(REGEX REPLACE "^#define BANK_LINES ([0-9]+)$" "\\1" PROFILE_BANK_LINES "${BANK_LINES_DECL}")
  if(PROFILE_BANK_LINES LESS RUNTIME_BANK_LINES)
    set(PARAMS_TARGET "${PROFILE}")
    set(RUNTIME_BANK_LINES "${PROFILE_BANK_LINES}")
  endif()
endforeach()
set(RUNTIME_PARAMS "${ISA_DIR}/${PARAMS_TARGET}")
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(ARTIFACT_DIR "${CMAKE_CURRENT_BINARY_DIR}/artifact" CACHE PATH "Compiled model artifact")
set(TOOLS "${COMPILER_BUILD}/bin/buddy-opt" "${COMPILER_BUILD}/bin/buddy-translate"
          "${COMPILER_BUILD}/bin/buddy-llc" "${BUDDY_SOURCE}/llvm/build/bin/opt")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CHIP_PB}")
function(compile_graph NAME SOURCE TARGET KIND)
  if(KIND STREQUAL "kernel")
    string(JOIN " " PASSES ${KERNEL_PASSES})
    set(IMPORT "\"${COMPILER_BUILD}/bin/buddy-opt\" \"${SOURCE}\" -pass-pipeline 'builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))'")
    if(KERNEL_TRACE_PASSES)
      string(JOIN " " TRACE_PASSES ${KERNEL_TRACE_PASSES})
      add_custom_command(OUTPUT ${NAME}.trace.mlir
        COMMAND bash -o pipefail -c
          "${IMPORT} | \"${COMPILER_BUILD}/bin/buddy-opt\" --target=${TARGET} ${TRACE_PASSES} -o ${NAME}.trace.mlir"
        DEPENDS ${SOURCE} ${COMPILER_BUILD}/bin/buddy-opt
        VERBATIM)
      set(LOWER_SOURCE ${NAME}.trace.mlir)
      set(LOWER "\"${COMPILER_BUILD}/bin/buddy-opt\" \"${LOWER_SOURCE}\" --target=${TARGET} ${PASSES}")
    else()
      set(LOWER_SOURCE ${SOURCE})
      set(LOWER "${IMPORT} | \"${COMPILER_BUILD}/bin/buddy-opt\" --target=${TARGET} ${PASSES}")
    endif()
  elseif(KIND STREQUAL "forward")
    string(JOIN " " PASSES ${FORWARD_PASSES})
    set(LOWER "\"${COMPILER_BUILD}/bin/buddy-opt\" \"${SOURCE}\" --target=${TARGET} ${PASSES}")
    set(LOWER_SOURCE ${SOURCE})
  else()
    message(FATAL_ERROR "Unknown graph kind: ${KIND}")
  endif()
  add_custom_command(OUTPUT ${NAME}.ll
    COMMAND bash -o pipefail -c
      "${LOWER} | \"${COMPILER_BUILD}/bin/buddy-translate\" --buddy-to-llvmir | \"${BUDDY_SOURCE}/llvm/build/bin/opt\" -passes='${LLVM_PASSES}' -o ${NAME}.ll -S"
    DEPENDS ${LOWER_SOURCE} ${TOOLS}
    VERBATIM)
  add_custom_command(OUTPUT ${NAME}.o
    COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${NAME}.ll
      -filetype=obj -mtriple=riscv64 -O${LLVM_OPT_LEVEL}
      -mattr=${MATTR} -float-abi=hard -o ${NAME}.o
    DEPENDS ${NAME}.ll ${COMPILER_BUILD}/bin/buddy-llc
    VERBATIM)
endfunction()
set(TRACE_SUPPORTED FALSE)
add_subdirectory("${MODEL_RECIPE}" "models/${MODEL}")
add_subdirectory("${REPO_ROOT}/stack/runtime" runtime)
if((CTRACE OR DTRACE) AND NOT TRACE_SUPPORTED)
  message(FATAL_ERROR "The selected model design does not support trace")
endif()
