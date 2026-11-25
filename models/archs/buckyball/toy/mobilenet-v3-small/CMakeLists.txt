# What the recipe compiles for this device config.
set(OUTPUT_DIR "${DESIGN_OUTPUT}/models/mobilenet-v3-small")

# ===----------------------------------------------------------------------===
# Import the FP32 graph (batch normalizations folded), the baremetal image
# and its reference.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/calibration.json
  COMMAND ${PYTHON} buddy-pytorch-calibrate.py ${OUTPUT_DIR}
  DEPENDS buddy-pytorch-calibrate.py
          quantized_model.py
          ${STACK_ROOT}/compiler/frontend/quant.py
          ${MODEL_DIR}/mobilenet-v3-small/data.py
          ${MODEL_DIR}/mobilenet-v3-small/model.py
          ${MODEL_DIR}/mobilenet-v3-small/output/mobilenet-v3-small/model.pth
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/01-import.txt
         ${OUTPUT_DIR}/subgraph0.mlir
         ${OUTPUT_DIR}/forward.mlir
         ${OUTPUT_DIR}/arg0.data
         ${OUTPUT_DIR}/arg1.data
         ${OUTPUT_DIR}/arg2.data
         ${OUTPUT_DIR}/inputs.data
         ${OUTPUT_DIR}/reference.data
  COMMAND ${PYTHON} buddy-pytorch-importer.py ${OUTPUT_DIR}
  DEPENDS ${OUTPUT_DIR}/calibration.json
          quantized_model.py
          frontend/permute/graph.py
          frontend/quant/graph.py
          ${COMPILER_FRONTEND_SOURCES}
          buddy-pytorch-importer.py
          ${MODEL_DIR}/mobilenet-v3-small/images/dog-32bit_224x224.bmp
          ${MODEL_DIR}/mobilenet-v3-small/model.py
          ${MODEL_DIR}/mobilenet-v3-small/output/mobilenet-v3-small/model.pth
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing MobileNetV3-Small"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The dataset of the accuracy runs: the ImageNet validation set and the
# logits of the FP32 model on it.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/dataset/pixels.data
         ${OUTPUT_DIR}/dataset/reference.data
         ${OUTPUT_DIR}/dataset/labels.data
  COMMAND ${PYTHON} buddy-pytorch-dataset.py ${OUTPUT_DIR}
  DEPENDS buddy-pytorch-dataset.py
          quantized_model.py
          ${OUTPUT_DIR}/calibration.json
          ${MODEL_DIR}/mobilenet-v3-small/model.py
          ${MODEL_DIR}/mobilenet-v3-small/data.py
          ${MODEL_DIR}/mobilenet-v3-small/output/mobilenet-v3-small/model.pth
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Preparing the ImageNet validation set of MobileNetV3-Small"
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
  COMMAND ${DESIGN_TARGET_OPT_toy} ${OUTPUT_DIR}/forward.mlir
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
          ${DESIGN_TARGET_PLUGIN_toy}
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
          ${LLVM_OPT}
  COMMENT "Building forward.o"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower the compute graph and export intermediate MLIR.
#
# Lower supported contractions into tiled VecBall operations on the toy hart.
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
  COMMAND ${DESIGN_TARGET_OPT_toy} ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_linalg.mlir
  COMMAND ${DESIGN_TARGET_OPT_toy} ${OUTPUT_DIR}/subgraph0_linalg.mlir
            -convert-linalg-to-acc
            -materialize-acc-results
            -o ${OUTPUT_DIR}/subgraph0_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_toy} ${OUTPUT_DIR}/subgraph0_acc.mlir
            -lower-frontend-fake-op
            -o ${OUTPUT_DIR}/subgraph0_frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_toy} ${OUTPUT_DIR}/subgraph0_frontend.mlir
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -canonicalize
            -optimize-allocation-liveness
            -o ${OUTPUT_DIR}/subgraph0_bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_toy} ${OUTPUT_DIR}/subgraph0_bufferized.mlir
            -assign-physical-banks
            -o ${OUTPUT_DIR}/subgraph0_physical.mlir
  COMMAND ${DESIGN_TARGET_OPT_toy} ${OUTPUT_DIR}/subgraph0_physical.mlir
            -llvm-request-c-wrappers
            -o ${OUTPUT_DIR}/subgraph0_cwrappers.mlir
  COMMAND ${DESIGN_TARGET_OPT_toy} ${OUTPUT_DIR}/subgraph0_cwrappers.mlir
            -lower-acc
            -o ${OUTPUT_DIR}/subgraph0_lowered.mlir
  COMMAND ${DESIGN_TARGET_OPT_toy} ${OUTPUT_DIR}/subgraph0_lowered.mlir
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
          ${DESIGN_TARGET_PLUGIN_toy}
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
          ${LLVM_OPT}
  COMMENT "Building subgraph0.o"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Link the model and runtime into a RISC-V shared library.
# ===----------------------------------------------------------------------===
add_library(mobilenet-v3-small-buckyball-toy-model SHARED EXCLUDE_FROM_ALL forward.o subgraph0.o)
set_target_properties(mobilenet-v3-small-buckyball-toy-model
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             LINKER_LANGUAGE CXX
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN;/lib"
)
target_link_libraries(mobilenet-v3-small-buckyball-toy-model
  PRIVATE runtime-linux-shared
)

# ===----------------------------------------------------------------------===
# Build the Buddy InferenceRunner plugin for RISC-V Linux.
# ===----------------------------------------------------------------------===
get_filename_component(FLATBUFFERS_HEADER "${TOOLCHAIN}/include/flatbuffers/flatbuffers.h" REALPATH)
get_filename_component(FLATBUFFERS_INCLUDE "${FLATBUFFERS_HEADER}/../.." ABSOLUTE)
add_library(mobilenet-v3-small-buckyball-toy-runner SHARED EXCLUDE_FROM_ALL runner.cpp)
set_target_properties(mobilenet-v3-small-buckyball-toy-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN;/lib"
)
target_include_directories(mobilenet-v3-small-buckyball-toy-runner
  PRIVATE ${LIB_DIR}
          ${FLATBUFFERS_INCLUDE}
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/runtime/include
          ${DESIGN_COMPILER}/runtime/include
)
target_link_libraries(mobilenet-v3-small-buckyball-toy-runner
  PRIVATE dl
)

# ===----------------------------------------------------------------------===
# Package the RAX: the RISC-V model and runner libraries and the FP32
# parameters (arg0.data).
# ===----------------------------------------------------------------------===
configure_file(mobilenet-v3-small.rhal.mlir.in ${OUTPUT_DIR}/mobilenet-v3-small.rhal.mlir @ONLY)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/mobilenet-v3-small.rax
  COMMAND ${COMPILER_BUILD}/bin/rax-pack ${OUTPUT_DIR}/mobilenet-v3-small.rhal.mlir
          --embed-payload
          -o ${OUTPUT_DIR}/mobilenet-v3-small.rax
  DEPENDS ${OUTPUT_DIR}/mobilenet-v3-small.rhal.mlir
          ${COMPILER_BUILD}/bin/rax-pack
          mobilenet-v3-small-buckyball-toy-model
          mobilenet-v3-small-buckyball-toy-runner
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
          ${OUTPUT_DIR}/arg2.data
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging mobilenet-v3-small for buckyball-toy"
  VERBATIM
)
set(BOOT_DIR "${STACK_ROOT}/runtime/boot")
list(TRANSFORM DESIGN_DEFS PREPEND "-D" OUTPUT_VARIABLE DESIGN_DEFINES)
list(
  APPEND DESIGN_DEFINES
  "-DDEVICE_TRACE_BASE=${DESIGN_DEVICE_TRACE_BASE}ULL"
  "-DDEVICE_TRACE_STRIDE=${DESIGN_DEVICE_TRACE_STRIDE}ULL"
)
list(TRANSFORM DESIGN_TARGET_DEFS_toy PREPEND "-D" OUTPUT_VARIABLE TARGET_DEFINES)
set(BAREMETAL_ELF "${DESIGN_OUTPUT}/workloads/models/toy-mobilenet-v3-small-baremetal")
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
          model.h
          ${LIB_DIR}/MemRef.h
          ${LIB_DIR}/testutils.h
          forward.o
          subgraph0.o
          ${BOOT_DIR}/start.S
          ${BOOT_DIR}/syscalls.c
          ${BOOT_DIR}/device.ld
          ${STACK_ROOT}/kernel/src/location.c
          ${RUNTIME_DIR}/services/mlir.cpp
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
set(DATASET_ELF "${DESIGN_OUTPUT}/workloads/models/toy-mobilenet-v3-small-dataset")
add_custom_command(
  OUTPUT ${DATASET_ELF}
  BYPRODUCTS dataset.o
             mlir-dataset.o
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DESIGN_OUTPUT}/workloads/models"
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++20 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -fno-builtin -I${LIB_DIR}
            --embed-dir=${OUTPUT_DIR} -c -DMODEL_IMAGE=${MODEL_IMAGE}ULL ${CMAKE_CURRENT_SOURCE_DIR}/dataset.cpp -o dataset.o
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++17 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -fno-builtin -I${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine -c ${RUNTIME_DIR}/services/mlir.cpp -o mlir-dataset.o
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -specs=nano.specs -specs=nosys.specs -nostartfiles -Wl,-T,${BOOT_DIR}/device.ld ${DESIGN_DEFINES} ${TARGET_DEFINES} -I${RUNTIME_DIR}/services -I${STACK_ROOT}/kernel/include ${BOOT_DIR}/start.S ${BOOT_DIR}/syscalls.c ${STACK_ROOT}/kernel/src/location.c ${RUNTIME_DIR}/trace/baremetal.c dataset.o mlir-dataset.o forward.o subgraph0.o -lm -lgcc -o ${DATASET_ELF}
  DEPENDS dataset.cpp
          model.h
          ${LIB_DIR}/ModelImage.h
          ${LIB_DIR}/MemRef.h
          ${LIB_DIR}/testutils.h
          forward.o
          subgraph0.o
          ${BOOT_DIR}/start.S
          ${BOOT_DIR}/syscalls.c
          ${BOOT_DIR}/device.ld
          ${STACK_ROOT}/kernel/src/location.c
          ${RUNTIME_DIR}/services/mlir.cpp
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
file(WRITE ${OUTPUT_DIR}/run.args
  "--model\nmobilenet-v3-small.rax\n--input\ninputs.data\n--reference\nreference.data\n"
)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/model.img
  COMMAND ${CMAKE_BINARY_DIR}/model-pack ${OUTPUT_DIR}/model.img ${OUTPUT_DIR} mobilenet-v3-small.rax inputs.data reference.data
            --mapped run.args
  DEPENDS model-pack-tool
          ${OUTPUT_DIR}/mobilenet-v3-small.rax
          ${OUTPUT_DIR}/run.args
          ${OUTPUT_DIR}/inputs.data
          ${OUTPUT_DIR}/reference.data
  VERBATIM
)
add_custom_target(
  mobilenet-v3-small-model-build
  DEPENDS
    ${OUTPUT_DIR}/mobilenet-v3-small.rax
    ${BAREMETAL_ELF}
    ${OUTPUT_DIR}/model.img
    ${DATASET_ELF}
    ${OUTPUT_DIR}/dataset/pixels.data
    ${OUTPUT_DIR}/dataset/reference.data
    ${OUTPUT_DIR}/dataset/labels.data
)

add_subdirectory(gvn)
