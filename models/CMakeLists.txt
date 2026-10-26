cmake_minimum_required(VERSION 3.22)
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

set(MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models")
set(LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib")

set(RUNTIME_DIR "${REPO_ROOT}/stack/runtime")
set(BUDDY_SOURCE "${REPO_ROOT}/stack/thirdparty/buddy-mlir")
set(COMPILER_BUILD "${BUDDY_SOURCE}/build")

# ===----------------------------------------------------------------------===
# Toolchain
# ===----------------------------------------------------------------------===
set(TOOLCHAIN "${REPO_ROOT}/result")
set(
  PYTHON
  ${CMAKE_COMMAND}
  -E
  env
  "PYTHONPATH=${REPO_ROOT}:${REPO_ROOT}/stack:${BUDDY_SOURCE}/llvm/build/tools/mlir/python_packages/mlir_core:${COMPILER_BUILD}/python_packages"
  "${TOOLCHAIN}/bin/python3"
)
# Programs of OS-invisible hardware threads are freestanding RISC-V images.
set(ELF_CC "${TOOLCHAIN}/bin/riscv64-unknown-elf-gcc")
set(ELF_OBJCOPY "${TOOLCHAIN}/bin/riscv64-unknown-elf-objcopy")

if(ARCH STREQUAL "nv" OR ARCH STREQUAL "x86")
  set(CMAKE_C_COMPILER "${TOOLCHAIN}/bin/gcc")
  set(CMAKE_CXX_COMPILER "${TOOLCHAIN}/bin/g++")
  add_compile_definitions(BUDDY_ENABLE_PNG)
else()
  set(CMAKE_C_COMPILER "${TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc")
  set(CMAKE_CXX_COMPILER "${TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++")
endif()

project(ModelBuild C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE COMPILER_FRONTEND_SOURCES CONFIGURE_DEPENDS "${REPO_ROOT}/stack/compiler/frontend/*.py")

# Packs the files of a model into the image loaded to the model region of the
# DDR. A host tool: it runs while building, whatever the target.
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/model-pack
  COMMAND g++ -std=c++20 -O2 -o ${CMAKE_BINARY_DIR}/model-pack ${CMAKE_CURRENT_SOURCE_DIR}/tools/model-pack.cpp
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/model-pack.cpp
  VERBATIM
)
add_custom_target(model-pack-tool DEPENDS ${CMAKE_BINARY_DIR}/model-pack)

if(ARCH)
  add_subdirectory(archs)
else()
  add_subdirectory(models)
endif()
