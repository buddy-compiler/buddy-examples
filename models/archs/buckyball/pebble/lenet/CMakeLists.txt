file(GLOB_RECURSE RECIPE_FRONTEND_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/frontend/*.py")

# What the recipe compiles for this device config; the calibration of the
# model belongs to the recipe.
set(OUTPUT_DIR "${DESIGN_OUTPUT}/models/lenet")
set(CALIBRATION "${CMAKE_CURRENT_SOURCE_DIR}/output/calibration.json")
set(PAYLOAD_DIR "${OUTPUT_DIR}/lenet.payload")

# ===----------------------------------------------------------------------===
# Calibrate the activation scales on training images.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CALIBRATION}
  COMMAND ${PYTHON} buddy-pytorch-calibrate.py
  DEPENDS buddy-pytorch-calibrate.py
          ${CMAKE_CURRENT_SOURCE_DIR}/../fixed_reference.py
          ${STACK_ROOT}/compiler/frontend/quant.py
          ${MODEL_DIR}/lenet/model.py
          ${MODEL_DIR}/lenet/output/lenet/model.pth
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Calibrating LeNet activations"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Import the LeNet graph, permute, quantize and partition it.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/01-import.txt
         ${OUTPUT_DIR}/02-permuted.txt
         ${OUTPUT_DIR}/03-quanted.txt
         ${OUTPUT_DIR}/04-fixed.txt
         ${OUTPUT_DIR}/05-partitioned.txt
         ${OUTPUT_DIR}/subgraph0.mlir
         ${OUTPUT_DIR}/forward.mlir
         ${OUTPUT_DIR}/arg0.data
         ${OUTPUT_DIR}/arg1.data
         ${OUTPUT_DIR}/arg2.data
         ${OUTPUT_DIR}/inputs.data
         ${OUTPUT_DIR}/reference.data
         ${OUTPUT_DIR}/dataset/inputs.data
         ${OUTPUT_DIR}/dataset/reference.data
         ${OUTPUT_DIR}/dataset/labels.data
  COMMAND ${PYTHON} buddy-pytorch-importer.py ${OUTPUT_DIR}
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          ${RECIPE_FRONTEND_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/../fixed.py
          ${CMAKE_CURRENT_SOURCE_DIR}/../fixed_reference.py
          buddy-pytorch-importer.py
          trace/trace.toml
          ${MODEL_DIR}/lenet/images/8.bmp
          ${CALIBRATION}
          ${MODEL_DIR}/lenet/model.py
          ${MODEL_DIR}/lenet/output/lenet/model.pth
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing LeNet"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower the model entry function to a RISC-V object.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward.o
  BYPRODUCTS ${OUTPUT_DIR}/forward.ll
             ${OUTPUT_DIR}/forward_opt.ll
             ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/forward.mlir
            -buffer-deallocation-simplification
            -convert-linalg-to-loops
            -eliminate-empty-tensors
            -llvm-request-c-wrappers
            -convert-math-to-llvm
            -convert-math-to-libm
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-arith-to-llvm
            -expand-strided-metadata
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/forward_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/forward.ll
  COMMAND ${LLVM_OPT} ${OUTPUT_DIR}/forward.ll -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=default<O2>" -S -o ${OUTPUT_DIR}/forward_opt.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward_opt.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o forward.o
  DEPENDS ${OUTPUT_DIR}/forward.mlir
          ${DESIGN_TARGET_PLUGIN_pebble}
          ${DESIGN_PB}
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
          ${LLVM_OPT}
  COMMENT "Building forward.o"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower the compute graph and export intermediate MLIR.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0.ll
             ${OUTPUT_DIR}/subgraph0_opt.ll
             ${OUTPUT_DIR}/subgraph0_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_acc.mlir
             ${OUTPUT_DIR}/subgraph0_frontend.mlir
             ${OUTPUT_DIR}/subgraph0_bufferized.mlir
             ${OUTPUT_DIR}/subgraph0_physical.mlir
             ${OUTPUT_DIR}/subgraph0_cwrappers.mlir
             ${OUTPUT_DIR}/subgraph0_lowered.mlir
             ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_linalg.mlir
  COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/subgraph0_linalg.mlir
            -convert-linalg-to-acc
            -o ${OUTPUT_DIR}/subgraph0_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/subgraph0_acc.mlir
            -lower-frontend-fake-op
            -o ${OUTPUT_DIR}/subgraph0_frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/subgraph0_frontend.mlir
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -linalg-fuse-elementwise-ops
            -canonicalize
            -cse
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -o ${OUTPUT_DIR}/subgraph0_bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/subgraph0_bufferized.mlir
            -cse
            -canonicalize
            -assign-physical-banks
            -o ${OUTPUT_DIR}/subgraph0_physical.mlir
  COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/subgraph0_physical.mlir
            -llvm-request-c-wrappers
            -o ${OUTPUT_DIR}/subgraph0_cwrappers.mlir
  COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/subgraph0_cwrappers.mlir
            -lower-acc
            -o ${OUTPUT_DIR}/subgraph0_lowered.mlir
  COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/subgraph0_lowered.mlir
            -expand-strided-metadata
            -convert-linalg-to-loops
            -canonicalize
            -cse
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/subgraph0.ll
  COMMAND ${LLVM_OPT} ${OUTPUT_DIR}/subgraph0.ll -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=default<O2>" -S -o ${OUTPUT_DIR}/subgraph0_opt.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0_opt.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o subgraph0.o
  DEPENDS ${OUTPUT_DIR}/subgraph0.mlir
          ${DESIGN_TARGET_PLUGIN_pebble}
          ${DESIGN_PB}
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
          ${LLVM_OPT}
  COMMENT "Building subgraph0.o"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Link the model and runtime into a RISC-V shared library.
# ===----------------------------------------------------------------------===
add_library(lenet-buckyball-pebble-model SHARED EXCLUDE_FROM_ALL forward.o subgraph0.o)
set_target_properties(lenet-buckyball-pebble-model
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             LINKER_LANGUAGE CXX
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN;/lib"
)
target_link_libraries(lenet-buckyball-pebble-model
  PRIVATE runtime-linux-shared
)

# ===----------------------------------------------------------------------===
# Build the Buddy InferenceRunner plugin for RISC-V Linux.
# ===----------------------------------------------------------------------===
get_filename_component(FLATBUFFERS_HEADER "${TOOLCHAIN}/include/flatbuffers/flatbuffers.h" REALPATH)
get_filename_component(FLATBUFFERS_INCLUDE "${FLATBUFFERS_HEADER}/../.." ABSOLUTE)
add_library(lenet-buckyball-pebble-runner SHARED EXCLUDE_FROM_ALL runner.cpp)
set_target_properties(lenet-buckyball-pebble-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN;/lib"
)
target_include_directories(lenet-buckyball-pebble-runner
  PRIVATE ${LIB_DIR}
          ${FLATBUFFERS_INCLUDE}
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/runtime/include
          ${DESIGN_COMPILER}/runtime/include
)
target_link_libraries(lenet-buckyball-pebble-runner
  PRIVATE dl
)

# ===----------------------------------------------------------------------===
# Package the RAX
#
# Contents:
# RISC-V model library (liblenet-buckyball-pebble-model.so),
# RISC-V runner library (liblenet-buckyball-pebble-runner.so),
# FP32 scales (arg0.data), INT32 biases (arg1.data), and
# INT8 weights (arg2.data).
#
# Output: output/lenet.rax embeds these files alongside RHAL metadata
# identifying the runner, model entry and parameter buffers.
# ===----------------------------------------------------------------------===
configure_file(lenet.rhal.mlir.in ${OUTPUT_DIR}/lenet.rhal.mlir @ONLY)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/lenet.rax
  COMMAND ${COMPILER_BUILD}/bin/rax-pack ${OUTPUT_DIR}/lenet.rhal.mlir
          --embed-payload
          -o ${OUTPUT_DIR}/lenet.rax
  DEPENDS ${OUTPUT_DIR}/lenet.rhal.mlir
          ${COMPILER_BUILD}/bin/rax-pack
          lenet-buckyball-pebble-model
          lenet-buckyball-pebble-runner
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
          ${OUTPUT_DIR}/arg2.data
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging lenet for buckyball-pebble"
  VERBATIM
)
set(BOOT_DIR "${STACK_ROOT}/runtime/boot")
list(TRANSFORM DESIGN_DEFS PREPEND "-D" OUTPUT_VARIABLE DESIGN_DEFINES)
list(
  APPEND DESIGN_DEFINES
  "-DDEVICE_TRACE_BASE=${DESIGN_DEVICE_TRACE_BASE}ULL"
  "-DDEVICE_TRACE_STRIDE=${DESIGN_DEVICE_TRACE_STRIDE}ULL"
)
list(TRANSFORM DESIGN_TARGET_DEFS_pebble PREPEND "-D" OUTPUT_VARIABLE TARGET_DEFINES)
set(BAREMETAL_ELF "${DESIGN_OUTPUT}/workloads/models/pebble-lenet-baremetal")
add_custom_command(
  OUTPUT ${BAREMETAL_ELF}
  BYPRODUCTS baremetal.o
             mlir-baremetal.o
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DESIGN_OUTPUT}/workloads/models"
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++20 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -fno-builtin -I${LIB_DIR}
            --embed-dir=${OUTPUT_DIR} -c ${CMAKE_CURRENT_SOURCE_DIR}/baremetal.cpp -o baremetal.o
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++17 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -fno-builtin -I${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine -c ${RUNTIME_DIR}/services/mlir.cpp -o mlir-baremetal.o
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -specs=nano.specs -specs=nosys.specs -nostartfiles -Wl,-T,${BOOT_DIR}/device.ld ${DESIGN_DEFINES} ${TARGET_DEFINES} -I${RUNTIME_DIR}/services -I${STACK_ROOT}/kernel/include ${BOOT_DIR}/start.S ${BOOT_DIR}/syscalls.c ${STACK_ROOT}/kernel/src/location.c ${RUNTIME_DIR}/trace/baremetal.c baremetal.o mlir-baremetal.o forward.o subgraph0.o -lm -lgcc -o ${BAREMETAL_ELF}
  DEPENDS baremetal.cpp
          ${STACK_ROOT}/kernel/src/location.c
          ${STACK_ROOT}/kernel/include/stream.h
          ${STACK_ROOT}/kernel/include/kernel.h
          ${RUNTIME_DIR}/trace/event.h
          ${RUNTIME_DIR}/services/mlir.cpp
          ${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine/CRunnerUtils.h
          ${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine/Msan.h
          model.h
          ${LIB_DIR}/MemRef.h
          ${LIB_DIR}/testutils.h
          forward.o
          subgraph0.o
          ${BOOT_DIR}/start.S
          ${BOOT_DIR}/syscalls.c
          ${BOOT_DIR}/device.ld
          ${RUNTIME_DIR}/trace/baremetal.c
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
          ${OUTPUT_DIR}/arg2.data
          ${OUTPUT_DIR}/inputs.data
          ${OUTPUT_DIR}/reference.data
          ${DESIGN_PB}
  COMMAND_EXPAND_LISTS
  VERBATIM
)

# The accuracy run (dataset.cpp): the samples of the model region, a shard
# of dataset/ that bench-device --bemu-accuracy packs into each run's DDR
# image.
math(EXPR MODEL_OFFSET "${DESIGN_DEVICE_DDR_BYTES} - ${DESIGN_DEVICE_DDR_MODEL}")
math(EXPR MODEL_IMAGE "${DESIGN_DEVICE_DDR_BASE} + ${MODEL_OFFSET}" OUTPUT_FORMAT HEXADECIMAL)
set(DATASET_ELF "${DESIGN_OUTPUT}/workloads/models/pebble-lenet-dataset")
add_custom_command(
  OUTPUT ${DATASET_ELF}
  BYPRODUCTS dataset.o
             mlir-dataset.o
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DESIGN_OUTPUT}/workloads/models"
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++20 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -fno-builtin -I${LIB_DIR} -DMODEL_IMAGE=${MODEL_IMAGE}ULL
            --embed-dir=${OUTPUT_DIR} -c ${CMAKE_CURRENT_SOURCE_DIR}/dataset.cpp -o dataset.o
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++17 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -fno-builtin -I${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine -c ${RUNTIME_DIR}/services/mlir.cpp -o mlir-dataset.o
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -specs=nano.specs -specs=nosys.specs -nostartfiles -Wl,-T,${BOOT_DIR}/device.ld ${DESIGN_DEFINES} ${TARGET_DEFINES} -I${RUNTIME_DIR}/services -I${STACK_ROOT}/kernel/include ${BOOT_DIR}/start.S ${BOOT_DIR}/syscalls.c ${STACK_ROOT}/kernel/src/location.c ${RUNTIME_DIR}/trace/baremetal.c dataset.o mlir-dataset.o forward.o subgraph0.o -lm -lgcc -o ${DATASET_ELF}
  DEPENDS dataset.cpp
          ${LIB_DIR}/ModelImage.h
          ${STACK_ROOT}/kernel/src/location.c
          ${STACK_ROOT}/kernel/include/stream.h
          ${STACK_ROOT}/kernel/include/kernel.h
          ${RUNTIME_DIR}/trace/event.h
          ${RUNTIME_DIR}/services/mlir.cpp
          ${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine/CRunnerUtils.h
          ${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine/Msan.h
          model.h
          ${LIB_DIR}/MemRef.h
          ${LIB_DIR}/testutils.h
          forward.o
          subgraph0.o
          ${BOOT_DIR}/start.S
          ${BOOT_DIR}/syscalls.c
          ${BOOT_DIR}/device.ld
          ${RUNTIME_DIR}/trace/baremetal.c
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
          ${OUTPUT_DIR}/arg2.data
          ${DESIGN_PB}
  COMMAND_EXPAND_LISTS
  VERBATIM
)

# Linux runs the RAX entry on the prepared image of the baremetal program and
# checks it against the same reference.
file(WRITE ${OUTPUT_DIR}/run.args "--model\nlenet.rax\n--input\ninputs.data\n--reference\nreference.data\n")
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/model.img
  COMMAND ${CMAKE_BINARY_DIR}/model-pack ${OUTPUT_DIR}/model.img ${OUTPUT_DIR} lenet.rax inputs.data reference.data
            --mapped run.args
  DEPENDS model-pack-tool
          ${OUTPUT_DIR}/lenet.rax
          ${OUTPUT_DIR}/run.args
          ${OUTPUT_DIR}/inputs.data
          ${OUTPUT_DIR}/reference.data
  VERBATIM
)
add_custom_target(
  lenet-model-build
  DEPENDS
    ${OUTPUT_DIR}/lenet.rax
    ${BAREMETAL_ELF}
    ${DATASET_ELF}
    ${OUTPUT_DIR}/model.img
    ${OUTPUT_DIR}/dataset/inputs.data
    ${OUTPUT_DIR}/dataset/reference.data
    ${OUTPUT_DIR}/dataset/labels.data
)

# The GVN slices of this model, which `bbdev gvn-bemu --debug` builds.
add_subdirectory(gvn)
