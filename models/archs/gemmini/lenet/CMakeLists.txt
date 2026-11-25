file(GLOB_RECURSE RECIPE_FRONTEND_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/frontend/*.py")
set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

add_custom_command(
  OUTPUT ${OUTPUT_DIR}/calibration.json
         ${OUTPUT_DIR}/check-images.data
         ${OUTPUT_DIR}/check-logits.data
         ${OUTPUT_DIR}/dataset/inputs.data
         ${OUTPUT_DIR}/dataset/reference.data
         ${OUTPUT_DIR}/dataset/labels.data
  COMMAND ${PYTHON} buddy-pytorch-calibrate.py
  DEPENDS ${STACK_ROOT}/compiler/frontend/quant.py
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-calibrate.py
          ${MODEL_DIR}/lenet/output/lenet/model.pth
          ${MODEL_DIR}/lenet/model.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Calibrating the lenet activations for gemmini"
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
  COMMAND ${PYTHON} buddy-pytorch-importer.py
  DEPENDS ${RECIPE_FRONTEND_SOURCES}
          ${COMPILER_FRONTEND_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${OUTPUT_DIR}/calibration.json
          ${MODEL_DIR}/lenet/output/lenet/model.pth
          ${MODEL_DIR}/lenet/model.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing lenet for gemmini"
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

add_library(lenet-gemmini-model STATIC EXCLUDE_FROM_ALL subgraph0.o forward.o)
set_target_properties(lenet-gemmini-model
  PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             LINKER_LANGUAGE CXX
)

add_library(lenet-gemmini-runner SHARED EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/runner.cpp)
set_target_properties(lenet-gemmini-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
)
add_dependencies(lenet-gemmini-runner lenet-gemmini-model)

target_include_directories(lenet-gemmini-runner
  PRIVATE ${BUDDY_SOURCE}/frontend/Interfaces
          ${LIB_DIR}
)
target_link_libraries(lenet-gemmini-runner
  PRIVATE lenet-gemmini-model
          gemmini-runtime
          model_image
)

# The runner owns inference and the backend's native scheduling lifecycle.
get_filename_component(FLATBUFFERS_HEADER "${TOOLCHAIN}/include/flatbuffers/flatbuffers.h" REALPATH)
get_filename_component(FLATBUFFERS_INCLUDE "${FLATBUFFERS_HEADER}/../.." ABSOLUTE)
target_include_directories(lenet-gemmini-runner
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
set_target_properties(lenet-gemmini-runner
  PROPERTIES POSITION_INDEPENDENT_CODE ON
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN;/lib"
)
configure_file(lenet.rhal.mlir.in ${OUTPUT_DIR}/lenet.rhal.mlir @ONLY)

# Package the runner, model parameters and input metadata in one RAX.
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/lenet.rax
  COMMAND ${COMPILER_BUILD}/bin/rax-pack ${OUTPUT_DIR}/lenet.rhal.mlir
          --embed-payload
          -o ${OUTPUT_DIR}/lenet.rax
  DEPENDS ${OUTPUT_DIR}/lenet.rhal.mlir
          ${COMPILER_BUILD}/bin/rax-pack
          lenet-gemmini-runner
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
          ${OUTPUT_DIR}/arg2.data
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging lenet for gemmini"
  VERBATIM
)
add_custom_target(lenet-model-build DEPENDS ${OUTPUT_DIR}/lenet.rax)

# The check of this recipe: the model as one program under spike with the
# Gemmini extension (pk), its predictions against the quantized model in
# software.
add_executable(lenet-gemmini-spike EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/spike.cpp)
set_target_properties(lenet-gemmini-spike
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
)
target_include_directories(lenet-gemmini-spike
  PRIVATE ${BUDDY_SOURCE}/frontend/Interfaces
)
target_link_libraries(lenet-gemmini-spike
  PRIVATE lenet-gemmini-model
          gemmini-runtime
          -static
)
add_custom_target(
  lenet-model-check
  COMMAND ${GEMMINI_SPIKE} pk $<TARGET_FILE:lenet-gemmini-spike> ${OUTPUT_DIR}
  DEPENDS
    lenet-gemmini-spike
    ${OUTPUT_DIR}/arg0.data
    ${OUTPUT_DIR}/arg1.data
    ${OUTPUT_DIR}/arg2.data
    ${OUTPUT_DIR}/check-images.data
    ${OUTPUT_DIR}/check-logits.data
  COMMENT "Running lenet on spike with the Gemmini extension"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The program of the device: the graph bare metal on Rocket with Gemmini (the
# Gemmini device's RTL), its parameters and the first four check images,
# against the quantized model's logits bit for bit.
# ===----------------------------------------------------------------------===
set(BAREMETAL_ELF "${DESIGN_OUTPUT}/workloads/models/gemmini-lenet-baremetal")
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
  COMMENT "Building lenet for the Gemmini device"
  COMMAND_EXPAND_LISTS
  VERBATIM
)

# The accuracy run (dataset.cpp): the MNIST test set of
# buddy-pytorch-calibrate.py, copied to models/lenet/dataset of the design,
# where bench-device --bemu-accuracy cuts it into the shards of the model
# region of each run's DDR image.
math(EXPR MODEL_OFFSET "${DESIGN_DEVICE_DDR_BYTES} - ${DESIGN_DEVICE_DDR_MODEL}")
math(EXPR MODEL_REGION "${DESIGN_DEVICE_DDR_BASE} + ${MODEL_OFFSET}" OUTPUT_FORMAT HEXADECIMAL)
set(DATASET_ELF "${DESIGN_OUTPUT}/workloads/models/gemmini-lenet-dataset")
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
  COMMENT "Building lenet for the Gemmini device"
  COMMAND_EXPAND_LISTS
  VERBATIM
)
add_custom_command(
  OUTPUT ${DESIGN_OUTPUT}/models/lenet/dataset/labels.data
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${OUTPUT_DIR}/dataset ${DESIGN_OUTPUT}/models/lenet/dataset
  DEPENDS ${OUTPUT_DIR}/dataset/inputs.data
          ${OUTPUT_DIR}/dataset/reference.data
          ${OUTPUT_DIR}/dataset/labels.data
  VERBATIM
)
add_custom_target(lenet-model-dataset DEPENDS ${DATASET_ELF} ${DESIGN_OUTPUT}/models/lenet/dataset/labels.data)
add_dependencies(lenet-model-build lenet-model-dataset)
add_custom_target(lenet-model-baremetal DEPENDS ${BAREMETAL_ELF})
add_dependencies(lenet-model-build lenet-model-baremetal)

add_custom_command(
  OUTPUT ${OUTPUT_DIR}/inputs.data
         ${OUTPUT_DIR}/reference.data
  COMMAND dd if=${OUTPUT_DIR}/check-images.data of=${OUTPUT_DIR}/inputs.data bs=12544 count=1 status=none
  COMMAND dd if=${OUTPUT_DIR}/check-logits.data of=${OUTPUT_DIR}/reference.data bs=160 count=1 status=none
  DEPENDS ${OUTPUT_DIR}/check-images.data
          ${OUTPUT_DIR}/check-logits.data
  VERBATIM
)
file(WRITE ${OUTPUT_DIR}/run.args "--model\nlenet.rax\n--input\ninputs.data\n--reference\nreference.data\n")
set(MODEL_IMAGE "${DESIGN_OUTPUT}/models/lenet/model.img")
add_custom_command(
  OUTPUT ${MODEL_IMAGE}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DESIGN_OUTPUT}/models/lenet"
  COMMAND ${CMAKE_BINARY_DIR}/model-pack ${MODEL_IMAGE} ${OUTPUT_DIR} lenet.rax inputs.data reference.data
            --mapped run.args
  DEPENDS model-pack-tool
          ${OUTPUT_DIR}/lenet.rax
          ${OUTPUT_DIR}/inputs.data
          ${OUTPUT_DIR}/reference.data
          ${OUTPUT_DIR}/run.args
  VERBATIM
)
add_custom_target(lenet-model-image DEPENDS ${MODEL_IMAGE})
add_dependencies(lenet-model-build lenet-model-image)

# The GVN slices of this model, which `bbdev gvn-bemu --debug` builds.
add_subdirectory(gvn)
