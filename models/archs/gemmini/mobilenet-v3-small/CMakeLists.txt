file(GLOB_RECURSE RECIPE_FRONTEND_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/frontend/*.py")
set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

add_custom_command(
  OUTPUT ${OUTPUT_DIR}/calibration.json
         ${OUTPUT_DIR}/check-images.data
         ${OUTPUT_DIR}/check-logits.data
         ${OUTPUT_DIR}/check-labels.data
  COMMAND ${PYTHON} buddy-pytorch-calibrate.py
  DEPENDS ${STACK_ROOT}/compiler/frontend/quant.py
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-calibrate.py
          ${MODEL_DIR}/mobilenet-v3-small/output/mobilenet-v3-small/model.pth
          ${MODEL_DIR}/mobilenet-v3-small/model.py
          ${MODEL_DIR}/mobilenet-v3-small/data.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Calibrating the mobilenet-v3-small activations for gemmini"
  VERBATIM
)

add_custom_command(
  OUTPUT ${OUTPUT_DIR}/01-import.txt
         ${OUTPUT_DIR}/02-permuted.txt
         ${OUTPUT_DIR}/03-quanted.txt
         ${OUTPUT_DIR}/forward.mlir
         ${OUTPUT_DIR}/subgraph0.mlir
         ${OUTPUT_DIR}/arg0.data
         ${OUTPUT_DIR}/arg1.data
         ${OUTPUT_DIR}/arg2.data
         ${OUTPUT_DIR}/Labels.txt
  COMMAND ${PYTHON} buddy-pytorch-importer.py
  DEPENDS ${RECIPE_FRONTEND_SOURCES}
          ${COMPILER_FRONTEND_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${OUTPUT_DIR}/calibration.json
          ${MODEL_DIR}/mobilenet-v3-small/output/mobilenet-v3-small/model.pth
          ${MODEL_DIR}/mobilenet-v3-small/model.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing mobilenet-v3-small for gemmini"
  VERBATIM
)

add_custom_command(
  OUTPUT forward.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -buffer-deallocation-simplification
            -convert-linalg-to-loops
            -eliminate-empty-tensors
            -llvm-request-c-wrappers
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-scf-to-cf
            -convert-arith-to-llvm
            -expand-strided-metadata
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -buffer-deallocation-simplification
            -convert-linalg-to-loops
            -eliminate-empty-tensors
            -llvm-request-c-wrappers
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-scf-to-cf
            -convert-arith-to-llvm
            -expand-strided-metadata
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${COMPILER_BUILD}/bin/buddy-translate
            --buddy-to-llvmir |
          ${COMPILER_BUILD}/bin/buddy-llc -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+m,+d,+xgemmini -float-abi=hard -o forward.o
  DEPENDS ${OUTPUT_DIR}/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM
)

add_custom_command(
  OUTPUT subgraph0.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_bufferized.mlir
             ${OUTPUT_DIR}/subgraph0_gemmini.mlir
             ${OUTPUT_DIR}/subgraph0_loops.mlir
             ${OUTPUT_DIR}/subgraph0_cwrappers.mlir
             ${OUTPUT_DIR}/subgraph0_lowered.mlir
             ${OUTPUT_DIR}/subgraph0_llvm.mlir
             ${OUTPUT_DIR}/subgraph0.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_linalg.mlir
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -gemmini-fuse-elementwise
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -o ${OUTPUT_DIR}/subgraph0_bufferized.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_bufferized.mlir
            -convert-linalg-to-gemmini
            -o ${OUTPUT_DIR}/subgraph0_gemmini.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_gemmini.mlir
            -batchmatmul-optimize
            -o ${OUTPUT_DIR}/subgraph0_loops.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_gemmini.mlir
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -o ${OUTPUT_DIR}/subgraph0_cwrappers.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_cwrappers.mlir
            -lower-gemmini
            -o ${OUTPUT_DIR}/subgraph0_lowered.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_lowered.mlir
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/subgraph0.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+m,+d,+xgemmini -float-abi=hard -o subgraph0.o
  DEPENDS ${OUTPUT_DIR}/subgraph0.mlir
  COMMENT "Building subgraph0.o"
  VERBATIM
)

add_library(mobilenet-v3-small-gemmini-model STATIC EXCLUDE_FROM_ALL subgraph0.o forward.o)
set_target_properties(mobilenet-v3-small-gemmini-model
  PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             LINKER_LANGUAGE CXX
)

add_library(mobilenet-v3-small-gemmini-runner SHARED EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/runner.cpp)
set_target_properties(mobilenet-v3-small-gemmini-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
)
add_dependencies(mobilenet-v3-small-gemmini-runner mobilenet-v3-small-gemmini-model)

target_include_directories(mobilenet-v3-small-gemmini-runner
  PRIVATE ${BUDDY_SOURCE}/frontend/Interfaces
          ${LIB_DIR}
)
target_link_libraries(mobilenet-v3-small-gemmini-runner
  PRIVATE mobilenet-v3-small-gemmini-model
          gemmini-runtime
          model_image
)

# The runner owns inference and the backend's native scheduling lifecycle.
get_filename_component(FLATBUFFERS_HEADER "${TOOLCHAIN}/include/flatbuffers/flatbuffers.h" REALPATH)
get_filename_component(FLATBUFFERS_INCLUDE "${FLATBUFFERS_HEADER}/../.." ABSOLUTE)
target_include_directories(mobilenet-v3-small-gemmini-runner
  PRIVATE ${LIB_DIR}
          ${FLATBUFFERS_INCLUDE}
          ${BUDDY_SOURCE}/runtime/include
          ${COMPILER_BUILD}/runtime/include
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/llvm/mlir/include
          ${OUTPUT_DIR}
          ${RUNTIME_DIR}/scheduler
          ${RUNTIME_DIR}/services
          ${RUNTIME_DIR}/../engine/include
)
set_target_properties(mobilenet-v3-small-gemmini-runner
  PROPERTIES POSITION_INDEPENDENT_CODE ON
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN;/lib"
)
configure_file(mobilenet-v3-small.rhal.mlir.in ${OUTPUT_DIR}/mobilenet-v3-small.rhal.mlir @ONLY)

# Package the runner, model parameters and input metadata in one RAX.
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/mobilenet-v3-small.rax
  COMMAND ${COMPILER_BUILD}/bin/rax-pack ${OUTPUT_DIR}/mobilenet-v3-small.rhal.mlir
          --embed-payload
          -o ${OUTPUT_DIR}/mobilenet-v3-small.rax
  DEPENDS ${OUTPUT_DIR}/mobilenet-v3-small.rhal.mlir
          ${COMPILER_BUILD}/bin/rax-pack
          mobilenet-v3-small-gemmini-runner
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
          ${OUTPUT_DIR}/arg2.data
          ${OUTPUT_DIR}/Labels.txt
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging mobilenet-v3-small for gemmini"
  VERBATIM
)
add_custom_target(mobilenet-v3-small-model-build DEPENDS ${OUTPUT_DIR}/mobilenet-v3-small.rax)

# The check of this recipe: the model as one program under spike with the
# Gemmini extension (pk), its predictions against the quantized model in
# software.
add_executable(mobilenet-v3-small-gemmini-spike EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/spike.cpp)
set_target_properties(mobilenet-v3-small-gemmini-spike
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
)
target_include_directories(mobilenet-v3-small-gemmini-spike
  PRIVATE ${BUDDY_SOURCE}/frontend/Interfaces
)
target_link_libraries(mobilenet-v3-small-gemmini-spike
  PRIVATE mobilenet-v3-small-gemmini-model
          gemmini-runtime
          -static
)
add_custom_target(
  mobilenet-v3-small-model-check
  COMMAND ${GEMMINI_SPIKE} pk $<TARGET_FILE:mobilenet-v3-small-gemmini-spike> ${OUTPUT_DIR}
  DEPENDS
    mobilenet-v3-small-gemmini-spike
    ${OUTPUT_DIR}/arg0.data
    ${OUTPUT_DIR}/arg1.data
    ${OUTPUT_DIR}/arg2.data
    ${OUTPUT_DIR}/check-images.data
    ${OUTPUT_DIR}/check-logits.data
    ${OUTPUT_DIR}/check-labels.data
  COMMENT "Running mobilenet-v3-small on spike with the Gemmini extension"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The program of the device: the graph bare metal on Rocket with Gemmini (the
# Gemmini device's RTL), its parameters and the first check image,
# against the quantized model's logits bit for bit.
# ===----------------------------------------------------------------------===
set(BAREMETAL_ELF "${DESIGN_OUTPUT}/workloads/models/gemmini-mobilenet-v3-small-baremetal")
add_custom_command(
  OUTPUT ${BAREMETAL_ELF}
  BYPRODUCTS baremetal.o
             mlir-baremetal.o
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DESIGN_OUTPUT}/workloads/models"
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++26 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -I${LIB_DIR}
            --embed-dir=${OUTPUT_DIR} -c ${CMAKE_CURRENT_SOURCE_DIR}/baremetal.cpp -o baremetal.o
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++17 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -fno-builtin -I${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine -c ${RUNTIME_DIR}/services/mlir.cpp -o mlir-baremetal.o
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -specs=nano.specs -specs=nosys.specs -nostartfiles -Wl,-T,${BOOT_DIR}/device.ld ${DESIGN_DEFINES} -I${RUNTIME_DIR}/services -I${STACK_ROOT}/kernel/include ${BOOT_DIR}/start.S ${BOOT_DIR}/syscalls.c ${STACK_ROOT}/kernel/src/location.c ${RUNTIME_DIR}/trace/baremetal.c baremetal.o mlir-baremetal.o forward.o subgraph0.o -lstdc++ -lm -lgcc -o ${BAREMETAL_ELF}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/baremetal.cpp
          forward.o
          subgraph0.o
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
          ${OUTPUT_DIR}/arg2.data
          ${OUTPUT_DIR}/check-images.data
          ${OUTPUT_DIR}/check-logits.data
          ${BOOT_DIR}/start.S
          ${BOOT_DIR}/syscalls.c
          ${BOOT_DIR}/device.ld
          ${STACK_ROOT}/kernel/src/location.c
          ${RUNTIME_DIR}/services/mlir.cpp
          ${RUNTIME_DIR}/trace/baremetal.c
          ${DESIGN_PB}
  COMMENT "Building mobilenet-v3-small for the Gemmini device"
  COMMAND_EXPAND_LISTS
  VERBATIM
)
add_custom_target(mobilenet-v3-small-model-baremetal DEPENDS ${BAREMETAL_ELF})
add_dependencies(mobilenet-v3-small-model-build mobilenet-v3-small-model-baremetal)

# ===----------------------------------------------------------------------===
# The slice: features 11 and 12, avgpool and classifier of the model,
# imported with the model's calibration, and its reference input and output
# (the logits) for the first check image.
# ===----------------------------------------------------------------------===
set(SLICE_DIR "${OUTPUT_DIR}/slice")
add_custom_command(
  OUTPUT ${SLICE_DIR}/forward.mlir
         ${SLICE_DIR}/subgraph0.mlir
         ${SLICE_DIR}/arg0.data
         ${SLICE_DIR}/arg1.data
         ${SLICE_DIR}/arg2.data
         ${SLICE_DIR}/check-input.data
         ${SLICE_DIR}/check-output.data
  COMMAND ${PYTHON} buddy-pytorch-slice.py
  DEPENDS ${RECIPE_FRONTEND_SOURCES}
          ${COMPILER_FRONTEND_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-slice.py
          ${OUTPUT_DIR}/calibration.json
          ${OUTPUT_DIR}/check-images.data
          ${OUTPUT_DIR}/check-logits.data
          ${MODEL_DIR}/mobilenet-v3-small/data.py
          ${MODEL_DIR}/mobilenet-v3-small/model.py
          ${MODEL_DIR}/mobilenet-v3-small/output/mobilenet-v3-small/model.pth
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing the mobilenet-v3-small slice for gemmini"
  VERBATIM
)

add_custom_command(
  OUTPUT slice-forward.o
  BYPRODUCTS ${SLICE_DIR}/forward_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${SLICE_DIR}/forward.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -buffer-deallocation-simplification
            -convert-linalg-to-loops
            -eliminate-empty-tensors
            -llvm-request-c-wrappers
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-scf-to-cf
            -convert-arith-to-llvm
            -expand-strided-metadata
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts
            -o ${SLICE_DIR}/forward_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${SLICE_DIR}/forward.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -buffer-deallocation-simplification
            -convert-linalg-to-loops
            -eliminate-empty-tensors
            -llvm-request-c-wrappers
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-scf-to-cf
            -convert-arith-to-llvm
            -expand-strided-metadata
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${COMPILER_BUILD}/bin/buddy-translate
            --buddy-to-llvmir |
          ${COMPILER_BUILD}/bin/buddy-llc -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+m,+d,+xgemmini -float-abi=hard -o slice-forward.o
  DEPENDS ${SLICE_DIR}/forward.mlir
  COMMENT "Building slice-forward.o"
  VERBATIM
)

add_custom_command(
  OUTPUT slice-subgraph0.o
  BYPRODUCTS ${SLICE_DIR}/subgraph0_linalg.mlir
             ${SLICE_DIR}/subgraph0_bufferized.mlir
             ${SLICE_DIR}/subgraph0_gemmini.mlir
             ${SLICE_DIR}/subgraph0_loops.mlir
             ${SLICE_DIR}/subgraph0_cwrappers.mlir
             ${SLICE_DIR}/subgraph0_lowered.mlir
             ${SLICE_DIR}/subgraph0_llvm.mlir
             ${SLICE_DIR}/subgraph0.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${SLICE_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))"
            -o ${SLICE_DIR}/subgraph0_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${SLICE_DIR}/subgraph0_linalg.mlir
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -gemmini-fuse-elementwise
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -o ${SLICE_DIR}/subgraph0_bufferized.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${SLICE_DIR}/subgraph0_bufferized.mlir
            -convert-linalg-to-gemmini
            -o ${SLICE_DIR}/subgraph0_gemmini.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${SLICE_DIR}/subgraph0_gemmini.mlir
            -batchmatmul-optimize
            -o ${SLICE_DIR}/subgraph0_loops.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${SLICE_DIR}/subgraph0_gemmini.mlir
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -o ${SLICE_DIR}/subgraph0_cwrappers.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${SLICE_DIR}/subgraph0_cwrappers.mlir
            -lower-gemmini
            -o ${SLICE_DIR}/subgraph0_lowered.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${SLICE_DIR}/subgraph0_lowered.mlir
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${SLICE_DIR}/subgraph0_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${SLICE_DIR}/subgraph0_llvm.mlir
            --buddy-to-llvmir -o ${SLICE_DIR}/subgraph0.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${SLICE_DIR}/subgraph0.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+m,+d,+xgemmini -float-abi=hard -o slice-subgraph0.o
  DEPENDS ${SLICE_DIR}/subgraph0.mlir
  COMMENT "Building slice-subgraph0.o"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# A slice of the model for the RTL of the device (buddy-pytorch-slice.py,
# slice.cpp): the tail of the model compiled as the model is, from the
# reference's features 10 output, against the model's logits bit for bit.
# ===----------------------------------------------------------------------===
set(BAREMETAL_ELF "${DESIGN_OUTPUT}/workloads/models/gemmini-mobilenet-v3-small-slice-baremetal")
add_custom_command(
  OUTPUT ${BAREMETAL_ELF}
  BYPRODUCTS slice.o
             slice-mlir.o
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DESIGN_OUTPUT}/workloads/models"
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++26 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti
            --embed-dir=${SLICE_DIR} -c ${CMAKE_CURRENT_SOURCE_DIR}/slice.cpp -o slice.o
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++17 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -fno-builtin -I${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine -c ${RUNTIME_DIR}/services/mlir.cpp -o slice-mlir.o
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -specs=nano.specs -specs=nosys.specs -nostartfiles -Wl,-T,${BOOT_DIR}/device.ld ${DESIGN_DEFINES} -I${RUNTIME_DIR}/services -I${STACK_ROOT}/kernel/include ${BOOT_DIR}/start.S ${BOOT_DIR}/syscalls.c ${STACK_ROOT}/kernel/src/location.c ${RUNTIME_DIR}/trace/baremetal.c slice.o slice-mlir.o slice-forward.o slice-subgraph0.o -lstdc++ -lm -lgcc -o ${BAREMETAL_ELF}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/slice.cpp
          slice-forward.o
          slice-subgraph0.o
          ${SLICE_DIR}/arg0.data
          ${SLICE_DIR}/arg1.data
          ${SLICE_DIR}/arg2.data
          ${SLICE_DIR}/check-input.data
          ${SLICE_DIR}/check-output.data
          ${BOOT_DIR}/start.S
          ${BOOT_DIR}/syscalls.c
          ${BOOT_DIR}/device.ld
          ${STACK_ROOT}/kernel/src/location.c
          ${RUNTIME_DIR}/services/mlir.cpp
          ${RUNTIME_DIR}/trace/baremetal.c
          ${DESIGN_PB}
  COMMENT "Building the mobilenet-v3-small slice for the Gemmini device"
  COMMAND_EXPAND_LISTS
  VERBATIM
)
add_custom_target(mobilenet-v3-small-slice-baremetal DEPENDS ${BAREMETAL_ELF})

add_custom_command(
  OUTPUT ${OUTPUT_DIR}/inputs.data
         ${OUTPUT_DIR}/reference.data
  COMMAND dd if=${OUTPUT_DIR}/check-images.data of=${OUTPUT_DIR}/inputs.data bs=602112 count=1 status=none
  COMMAND dd if=${OUTPUT_DIR}/check-logits.data of=${OUTPUT_DIR}/reference.data bs=4000 count=1 status=none
  DEPENDS ${OUTPUT_DIR}/check-images.data
          ${OUTPUT_DIR}/check-logits.data
  VERBATIM
)
file(WRITE ${OUTPUT_DIR}/run.args
  "--model\nmobilenet-v3-small.rax\n--input\ninputs.data\n--reference\nreference.data\n"
)
set(MODEL_IMAGE "${DESIGN_OUTPUT}/models/mobilenet-v3-small/model.img")
add_custom_command(
  OUTPUT ${MODEL_IMAGE}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DESIGN_OUTPUT}/models/mobilenet-v3-small"
  COMMAND ${CMAKE_BINARY_DIR}/model-pack ${MODEL_IMAGE} ${OUTPUT_DIR} mobilenet-v3-small.rax inputs.data reference.data
            --mapped run.args
  DEPENDS model-pack-tool
          ${OUTPUT_DIR}/mobilenet-v3-small.rax
          ${OUTPUT_DIR}/inputs.data
          ${OUTPUT_DIR}/reference.data
          ${OUTPUT_DIR}/run.args
  VERBATIM
)
add_custom_target(mobilenet-v3-small-model-image DEPENDS ${MODEL_IMAGE})
add_dependencies(mobilenet-v3-small-model-build mobilenet-v3-small-model-image)

# The accuracy run (dataset.cpp): the ImageNet validation set in
# models/mobilenet-v3-small/dataset of the design, where bench-device --bemu-accuracy
# cuts it into the shards of the model region of each run's DDR image.
add_custom_command(
  OUTPUT ${DESIGN_OUTPUT}/models/mobilenet-v3-small/dataset/pixels.data
         ${DESIGN_OUTPUT}/models/mobilenet-v3-small/dataset/reference.data
         ${DESIGN_OUTPUT}/models/mobilenet-v3-small/dataset/labels.data
  COMMAND ${PYTHON} buddy-pytorch-dataset.py ${DESIGN_OUTPUT}/models/mobilenet-v3-small
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-dataset.py
          ${STACK_ROOT}/compiler/frontend/quant.py
          ${OUTPUT_DIR}/calibration.json
          ${MODEL_DIR}/mobilenet-v3-small/model.py
          ${MODEL_DIR}/mobilenet-v3-small/data.py
          ${MODEL_DIR}/mobilenet-v3-small/output/mobilenet-v3-small/model.pth
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Preparing the ImageNet validation set of mobilenet-v3-small for gemmini"
  VERBATIM
)
math(EXPR MODEL_OFFSET "${DESIGN_DEVICE_DDR_BYTES} - ${DESIGN_DEVICE_DDR_MODEL}")
math(EXPR MODEL_REGION "${DESIGN_DEVICE_DDR_BASE} + ${MODEL_OFFSET}" OUTPUT_FORMAT HEXADECIMAL)
set(DATASET_ELF "${DESIGN_OUTPUT}/workloads/models/gemmini-mobilenet-v3-small-dataset")
add_custom_command(
  OUTPUT ${DATASET_ELF}
  BYPRODUCTS dataset.o
             mlir-dataset.o
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DESIGN_OUTPUT}/workloads/models"
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++26 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -I${LIB_DIR}
            --embed-dir=${OUTPUT_DIR} -DMODEL_REGION=${MODEL_REGION}ULL -c ${CMAKE_CURRENT_SOURCE_DIR}/dataset.cpp -o dataset.o
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++17 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -fno-builtin -I${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine -c ${RUNTIME_DIR}/services/mlir.cpp -o mlir-dataset.o
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -specs=nano.specs -specs=nosys.specs -nostartfiles -Wl,-T,${BOOT_DIR}/device.ld ${DESIGN_DEFINES} -I${RUNTIME_DIR}/services -I${STACK_ROOT}/kernel/include ${BOOT_DIR}/start.S ${BOOT_DIR}/syscalls.c ${STACK_ROOT}/kernel/src/location.c ${RUNTIME_DIR}/trace/baremetal.c dataset.o mlir-dataset.o forward.o subgraph0.o -lstdc++ -lm -lgcc -o ${DATASET_ELF}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/dataset.cpp
          ${LIB_DIR}/ModelImage.h
          forward.o
          subgraph0.o
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
          ${OUTPUT_DIR}/arg2.data
          ${BOOT_DIR}/start.S
          ${BOOT_DIR}/syscalls.c
          ${BOOT_DIR}/device.ld
          ${STACK_ROOT}/kernel/src/location.c
          ${RUNTIME_DIR}/services/mlir.cpp
          ${RUNTIME_DIR}/trace/baremetal.c
          ${DESIGN_PB}
  COMMENT "Building the mobilenet-v3-small accuracy run for the Gemmini device"
  COMMAND_EXPAND_LISTS
  VERBATIM
)
add_custom_target(
  mobilenet-v3-small-model-dataset
  DEPENDS ${DATASET_ELF} ${DESIGN_OUTPUT}/models/mobilenet-v3-small/dataset/labels.data
)
add_dependencies(mobilenet-v3-small-model-build mobilenet-v3-small-model-dataset)

# The GVN slices of this model, which `bbdev gvn-bemu --debug` builds.
add_subdirectory(gvn)
