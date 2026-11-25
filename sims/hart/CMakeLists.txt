# ===----------------------------------------------------------------------===
# The software of the hart simulation's system (arch: sims/hart): one hart
# under test, with its hardware threads, in a single-hart system whose
# peripherals sit at the simulator's fixed addresses. It knows nothing of the
# SoC of any design; that belongs to the design's device stack.
#
# The hart simulation entry of a design builds it after the hart's stack,
# configured with DESIGN_CMAKE and the RISC-V Linux toolchain. Products,
# exported in device.cmake for the programs that run in the simulation:
#   runtime, runtime-baremetal, runtime-linux  job scheduling over the hart
#   kernel                                     hardware threads and instruction streams
#   bindings                                   how the hart runs jobs and each
#                                              hardware thread is loaded
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(hart-simulation-system LANGUAGES C CXX)
include("${DESIGN_CMAKE}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(LENGTH DESIGN_HARTS HART_COUNT)
if(NOT HART_COUNT EQUAL 1)
  message(FATAL_ERROR "the hart simulation holds one hart, got: ${DESIGN_HARTS}")
endif()

# How the hart runs jobs, and how each of its hardware threads is loaded.
# Nothing references these bindings, so they are linked as objects rather than
# taken from an archive; as link options they come before the libraries they call.
add_library(bindings INTERFACE)
target_link_options(bindings INTERFACE "${DESIGN_HART_BUILD_${DESIGN_HARTS}}/execute.o")
foreach(THREAD IN LISTS DESIGN_THREAD_STACKS)
  target_link_options(bindings INTERFACE "${DESIGN_THREAD_BUILD_${THREAD}}/loader.o")
endforeach()

add_subdirectory("${DESIGN_REPO}/stack/kernel" kernel)
add_subdirectory("${DESIGN_REPO}/stack/runtime" runtime)
export(
  TARGETS bindings runtime runtime-baremetal runtime-linux kernel runtime-linux-shared kernel-linux kernel-shared
  FILE device.cmake
)
