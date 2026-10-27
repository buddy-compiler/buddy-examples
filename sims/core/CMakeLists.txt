# ===----------------------------------------------------------------------===
# The software of the core simulation's system (arch: sims/core): one core
# under test, with its hardware threads, in a single-core system whose
# peripherals sit at the simulator's fixed addresses. It knows nothing of the
# SoC of any design; that belongs to the design's device stack.
#
# The core simulation entry of a design builds it after the core's stack,
# configured with DESIGN_CMAKE and the RISC-V Linux toolchain. Products,
# exported in device.cmake for the programs that run in the simulation:
#   runtime, runtime-baremetal, runtime-linux  job scheduling over the core
#   engine                                     hardware threads and instruction streams
#   bindings                                   how the core runs jobs and each
#                                              hardware thread is loaded
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(core-simulation-system LANGUAGES C CXX)
include("${DESIGN_CMAKE}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(LENGTH DESIGN_CORES CORE_COUNT)
if(NOT CORE_COUNT EQUAL 1)
  message(FATAL_ERROR "the core simulation holds one core, got: ${DESIGN_CORES}")
endif()

# How the core runs jobs, and how each of its hardware threads is loaded.
# Nothing references these bindings, so they are linked as objects rather than
# taken from an archive; as link options they come before the libraries they call.
add_library(bindings INTERFACE)
target_link_options(bindings INTERFACE "${DESIGN_CORE_BUILD_${DESIGN_CORES}}/execute.o")
foreach(THREAD IN LISTS DESIGN_THREAD_STACKS)
  target_link_options(bindings INTERFACE "${DESIGN_THREAD_BUILD_${THREAD}}/loader.o")
endforeach()
# DMA helpers are sized by the target with the fewest bank lines.
list(GET DESIGN_RUNTIME_TARGETS 0 PARAMS_TARGET)
set(RUNTIME_DEFS ${DESIGN_TARGET_DEFS_${PARAMS_TARGET}})

add_subdirectory("${DESIGN_REPO}/stack/engine" engine)
add_subdirectory("${DESIGN_REPO}/stack/runtime" runtime)
export(
  TARGETS bindings runtime runtime-baremetal runtime-linux engine runtime-shared runtime-linux-shared engine-shared
  FILE device.cmake
)
