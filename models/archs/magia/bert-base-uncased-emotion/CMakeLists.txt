set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

# The mesh's FP16 fused multiply-adds on this machine's AVX512-FP16: the
# products of the reference of the import and of the check.
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/libmagia-gemm.so
  COMMAND ${TOOLCHAIN}/bin/gcc -O3 -mavx512fp16 -mavx512vl -shared -fPIC ${MAGIA_RUNTIME}/reference/gemm.c -o ${OUTPUT_DIR}/libmagia-gemm.so
  DEPENDS ${MAGIA_RUNTIME}/reference/gemm.c
  COMMENT "Building the mesh's products in software"
  VERBATIM
)

add_custom_command(
  OUTPUT ${OUTPUT_DIR}/forward.mlir
         ${OUTPUT_DIR}/subgraph0.mlir
         ${OUTPUT_DIR}/arg0.data
         ${OUTPUT_DIR}/arg1.data
         ${OUTPUT_DIR}/check-ids.data
         ${OUTPUT_DIR}/check-types.data
         ${OUTPUT_DIR}/check-mask.data
         ${OUTPUT_DIR}/check-positions.data
         ${OUTPUT_DIR}/check-logits.data
         ${OUTPUT_DIR}/slice-x.data
         ${OUTPUT_DIR}/slice-w.data
         ${OUTPUT_DIR}/slice-z.data
         ${OUTPUT_DIR}/dataset/ids.data
         ${OUTPUT_DIR}/dataset/positions.data
         ${OUTPUT_DIR}/dataset/types.data
         ${OUTPUT_DIR}/dataset/masks.data
         ${OUTPUT_DIR}/dataset/reference.data
         ${OUTPUT_DIR}/dataset/labels.data
  COMMAND ${PYTHON} buddy-pytorch-importer.py ${OUTPUT_DIR}/libmagia-gemm.so
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          ${OUTPUT_DIR}/libmagia-gemm.so
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${CMAKE_CURRENT_SOURCE_DIR}/frontend/weights.py
          ${MODEL_DIR}/bert-base-uncased-emotion/model.py
          ${MODEL_DIR}/bert-base-uncased-emotion/data.py
          ${MODEL_DIR}/bert-base-uncased-emotion/output/bert-base-uncased-emotion/config.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing bert-base-uncased-emotion for magia"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The entry: the parameters of the subgraph, by view.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/forward_llvm.mlir
  BYPRODUCTS ${OUTPUT_DIR}/forward_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/forward_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_linalg.mlir
            -convert-elementwise-to-linalg
            -convert-linalg-to-loops
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/forward_llvm.mlir
  DEPENDS ${OUTPUT_DIR}/forward.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
  COMMENT "Lowering the bert-base-uncased-emotion entry for magia"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The graph: its matrix products to the mesh, the rest to the host hart.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/subgraph0_llvm.mlir
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_bufferized.mlir
             ${OUTPUT_DIR}/subgraph0_magia.mlir
             ${OUTPUT_DIR}/subgraph0_lowered.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_linalg.mlir
            -convert-elementwise-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-bufferization-to-memref
            -llvm-request-c-wrappers
            -o ${OUTPUT_DIR}/subgraph0_bufferized.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_bufferized.mlir
            -convert-linalg-to-magia
            -o ${OUTPUT_DIR}/subgraph0_magia.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_magia.mlir
            -lower-magia
            -o ${OUTPUT_DIR}/subgraph0_lowered.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_lowered.mlir
            -convert-linalg-to-loops
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/subgraph0_llvm.mlir
  DEPENDS ${OUTPUT_DIR}/subgraph0.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
  COMMENT "Lowering the bert-base-uncased-emotion graph for magia"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The check of this recipe: the model compiled for this machine, the mesh's
# products in software, its predictions against the model in PyTorch with the
# same products.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/bert-base-uncased-emotion-magia-check
  BYPRODUCTS ${OUTPUT_DIR}/check-forward.o
             ${OUTPUT_DIR}/check-subgraph0.o
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/forward_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/forward.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward.ll -filetype=obj -relocation-model=pic -O2 -o ${OUTPUT_DIR}/check-forward.o
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/subgraph0.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0.ll -filetype=obj -relocation-model=pic -O2 -o ${OUTPUT_DIR}/check-subgraph0.o
  COMMAND ${TOOLCHAIN}/bin/g++ -std=c++20 -O2 -I${BUDDY_SOURCE}/frontend/Interfaces ${CMAKE_CURRENT_SOURCE_DIR}/check.cpp ${OUTPUT_DIR}/check-forward.o ${OUTPUT_DIR}/check-subgraph0.o ${OUTPUT_DIR}/libmagia-gemm.so -Wl,-rpath,${OUTPUT_DIR} -L${BUDDY_SOURCE}/llvm/build/lib -Wl,-rpath,${BUDDY_SOURCE}/llvm/build/lib -lmlir_c_runner_utils -lmlir_runner_utils -lm -o ${OUTPUT_DIR}/bert-base-uncased-emotion-magia-check
  DEPENDS ${OUTPUT_DIR}/forward_llvm.mlir
          ${OUTPUT_DIR}/subgraph0_llvm.mlir
          ${CMAKE_CURRENT_SOURCE_DIR}/check.cpp
          ${OUTPUT_DIR}/libmagia-gemm.so
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Building the bert-base-uncased-emotion check for magia"
  VERBATIM
)
add_custom_target(
  bert-base-uncased-emotion-model-check
  COMMAND ${OUTPUT_DIR}/bert-base-uncased-emotion-magia-check ${OUTPUT_DIR}
  DEPENDS
    ${OUTPUT_DIR}/bert-base-uncased-emotion-magia-check
    ${OUTPUT_DIR}/arg0.data
    ${OUTPUT_DIR}/arg1.data
    ${OUTPUT_DIR}/check-ids.data
    ${OUTPUT_DIR}/check-types.data
    ${OUTPUT_DIR}/check-mask.data
    ${OUTPUT_DIR}/check-positions.data
    ${OUTPUT_DIR}/check-logits.data
  COMMENT "Running bert-base-uncased-emotion for magia with the mesh in software"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The program of the device: the graph for Loom (RV64GC with half precision), its parameters, the
# check sentences, the firmware of the tiles and Loom's runtime of the mesh
# (the products of the hart's stack), laid out by the device's device.ld.
# ===----------------------------------------------------------------------===
set(BAREMETAL_ELF "${DESIGN_OUTPUT}/workloads/models/magia-bert-base-uncased-emotion-baremetal")
add_custom_command(
  OUTPUT forward.o
         subgraph0.o
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/forward_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/forward.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c,+zfh -target-abi=lp64d -code-model=medium -O2 -o forward.o
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/subgraph0.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c,+zfh -target-abi=lp64d -code-model=medium -O2 -o subgraph0.o
  DEPENDS ${OUTPUT_DIR}/forward_llvm.mlir
          ${OUTPUT_DIR}/subgraph0_llvm.mlir
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  VERBATIM
)
add_custom_command(
  OUTPUT ${BAREMETAL_ELF}
  BYPRODUCTS baremetal.o
             checked.o
             mlir-baremetal.o
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DESIGN_OUTPUT}/workloads/models"
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++26 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -I${LIB_DIR}
            --embed-dir=${OUTPUT_DIR} -c ${CMAKE_CURRENT_SOURCE_DIR}/baremetal.cpp -o baremetal.o
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++26 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -I${LIB_DIR}
            --embed-dir=${OUTPUT_DIR} -c ${CMAKE_CURRENT_SOURCE_DIR}/checked.cpp -o checked.o
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++17 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -fno-builtin -I${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine -c ${RUNTIME_DIR}/services/mlir.cpp -o mlir-baremetal.o
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -specs=nano.specs -specs=nosys.specs -nostartfiles -Wl,-T,${DESIGN_DEVICE_STACK}/device.ld ${DESIGN_DEFINES} ${BOOT_DIR}/start.S ${BOOT_DIR}/syscalls.c baremetal.o checked.o mlir-baremetal.o forward.o subgraph0.o ${DESIGN_HART_BUILD_magia}/gemm.o ${DESIGN_HART_BUILD_magia}/mesh.o -lstdc++ -lm -o ${BAREMETAL_ELF}
  DEPENDS forward.o
          subgraph0.o
          checked.h
          checked.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/baremetal.cpp
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
          ${OUTPUT_DIR}/check-ids.data
          ${OUTPUT_DIR}/check-types.data
          ${OUTPUT_DIR}/check-mask.data
          ${OUTPUT_DIR}/check-positions.data
          ${OUTPUT_DIR}/check-logits.data
          ${BOOT_DIR}/start.S
          ${BOOT_DIR}/syscalls.c
          ${DESIGN_DEVICE_STACK}/device.ld
          ${RUNTIME_DIR}/services/mlir.cpp
          ${DESIGN_HART_BUILD_magia}/gemm.o
          ${DESIGN_HART_BUILD_magia}/mesh.o
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
          ${DESIGN_PB}
  COMMENT "Building bert-base-uncased-emotion for the MAGIA device"
  COMMAND_EXPAND_LISTS
  VERBATIM
)

# The accuracy run (dataset.cpp): the validation sentences of the importer,
# copied to models/bert-base-uncased-emotion/dataset of the design, where
# bench-device --bemu-accuracy cuts it into the shards of the model region
# of each run's DDR image.
set(DATASET_ELF "${DESIGN_OUTPUT}/workloads/models/magia-bert-base-uncased-emotion-dataset")
add_custom_command(
  OUTPUT ${DATASET_ELF}
  BYPRODUCTS dataset.o
             mlir-dataset.o
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DESIGN_OUTPUT}/workloads/models"
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++26 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -I${LIB_DIR}
            --embed-dir=${OUTPUT_DIR} -DMODEL_IMAGE=${MODEL_IMAGE}ULL -c ${CMAKE_CURRENT_SOURCE_DIR}/dataset.cpp -o dataset.o
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++17 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -fno-builtin -I${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine -c ${RUNTIME_DIR}/services/mlir.cpp -o mlir-dataset.o
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -specs=nano.specs -specs=nosys.specs -nostartfiles -Wl,-T,${DESIGN_DEVICE_STACK}/device.ld ${DESIGN_DEFINES} ${BOOT_DIR}/start.S ${BOOT_DIR}/syscalls.c dataset.o mlir-dataset.o forward.o subgraph0.o ${DESIGN_HART_BUILD_magia}/gemm.o ${DESIGN_HART_BUILD_magia}/mesh.o -lstdc++ -lm -o ${DATASET_ELF}
  DEPENDS forward.o
          subgraph0.o
          ${CMAKE_CURRENT_SOURCE_DIR}/dataset.cpp
          ${LIB_DIR}/ModelImage.h
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
          ${BOOT_DIR}/start.S
          ${BOOT_DIR}/syscalls.c
          ${DESIGN_DEVICE_STACK}/device.ld
          ${RUNTIME_DIR}/services/mlir.cpp
          ${DESIGN_HART_BUILD_magia}/gemm.o
          ${DESIGN_HART_BUILD_magia}/mesh.o
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
          ${DESIGN_PB}
  COMMENT "Building the bert-base-uncased-emotion accuracy run for the MAGIA device"
  COMMAND_EXPAND_LISTS
  VERBATIM
)
add_custom_command(
  OUTPUT ${DESIGN_OUTPUT}/models/bert-base-uncased-emotion/dataset/labels.data
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${OUTPUT_DIR}/dataset ${DESIGN_OUTPUT}/models/bert-base-uncased-emotion/dataset
  DEPENDS ${OUTPUT_DIR}/dataset/ids.data
          ${OUTPUT_DIR}/dataset/positions.data
          ${OUTPUT_DIR}/dataset/types.data
          ${OUTPUT_DIR}/dataset/masks.data
          ${OUTPUT_DIR}/dataset/reference.data
          ${OUTPUT_DIR}/dataset/labels.data
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The model under Linux: the same graph and check in a runner of buddy-cli.
# The weights and parameters are files of the model image, mapped from the
# DDR; the mesh reads the weights there (the Linux runtime of the hart's
# stack copies the other matrices of a product into the DDR it keeps).
# ===----------------------------------------------------------------------===
get_filename_component(FLATBUFFERS_HEADER "${TOOLCHAIN}/include/flatbuffers/flatbuffers.h" REALPATH)
get_filename_component(FLATBUFFERS_INCLUDE "${FLATBUFFERS_HEADER}/../.." ABSOLUTE)
add_library(bert-base-uncased-emotion-magia-runner SHARED EXCLUDE_FROM_ALL runner.cpp checked.cpp forward.o
  subgraph0.o ${DESIGN_HART_BUILD_magia}/gemm-linux.o ${DESIGN_HART_BUILD_magia}/firmware-linux.o
)
target_compile_options(bert-base-uncased-emotion-magia-runner PRIVATE --embed-dir=${OUTPUT_DIR})
target_include_directories(bert-base-uncased-emotion-magia-runner
  PRIVATE ${LIB_DIR}
          ${RUNTIME_DIR}/services
          ${FLATBUFFERS_INCLUDE}
          ${BUDDY_SOURCE}/runtime/include
          ${COMPILER_BUILD}/runtime/include
)
target_link_libraries(bert-base-uncased-emotion-magia-runner
  PRIVATE magia-runtime
          m
)
set_target_properties(bert-base-uncased-emotion-magia-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             CXX_STANDARD 26
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN;/lib"
)
configure_file(bert-base-uncased-emotion.rhal.mlir.in ${OUTPUT_DIR}/bert-base-uncased-emotion.rhal.mlir @ONLY)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/bert-base-uncased-emotion.rax
  COMMAND ${COMPILER_BUILD}/bin/rax-pack ${OUTPUT_DIR}/bert-base-uncased-emotion.rhal.mlir
          --embed-payload
          -o ${OUTPUT_DIR}/bert-base-uncased-emotion.rax
  DEPENDS ${OUTPUT_DIR}/bert-base-uncased-emotion.rhal.mlir
          bert-base-uncased-emotion-magia-runner
          ${COMPILER_BUILD}/bin/rax-pack
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)
file(WRITE ${OUTPUT_DIR}/run.args "--model\nbert-base-uncased-emotion.rax\n--input\ncheck-ids.data\n")
set(MODEL_IMAGE_FILE "${DESIGN_OUTPUT}/models/bert-base-uncased-emotion/model.img")
add_custom_command(
  OUTPUT ${MODEL_IMAGE_FILE}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DESIGN_OUTPUT}/models/bert-base-uncased-emotion"
  COMMAND ${CMAKE_BINARY_DIR}/model-pack ${MODEL_IMAGE_FILE} ${OUTPUT_DIR} bert-base-uncased-emotion.rax check-ids.data
            --mapped run.args
            --mapped arg0.data
            --mapped arg1.data
  DEPENDS model-pack-tool
          ${OUTPUT_DIR}/bert-base-uncased-emotion.rax
          ${OUTPUT_DIR}/run.args
          ${OUTPUT_DIR}/check-ids.data
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
  VERBATIM
)
add_custom_target(
  bert-base-uncased-emotion-model-build
  DEPENDS
    ${BAREMETAL_ELF}
    ${MODEL_IMAGE_FILE}
    ${DATASET_ELF}
    ${DESIGN_OUTPUT}/models/bert-base-uncased-emotion/dataset/labels.data
)

# ===----------------------------------------------------------------------===
# A slice of the model for the RTL of the device (slice.cpp): the first
# product, on the mesh, against the reference run's values.
# ===----------------------------------------------------------------------===
set(SLICE_ELF "${DESIGN_OUTPUT}/workloads/models/magia-bert-base-uncased-emotion-slice-baremetal")
add_custom_command(
  OUTPUT ${SLICE_ELF}
  BYPRODUCTS slice.o
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DESIGN_OUTPUT}/workloads/models"
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++26 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti
            --embed-dir=${OUTPUT_DIR} -c ${CMAKE_CURRENT_SOURCE_DIR}/slice.cpp -o slice.o
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -specs=nano.specs -specs=nosys.specs -nostartfiles -Wl,-T,${DESIGN_DEVICE_STACK}/device.ld ${DESIGN_DEFINES} ${BOOT_DIR}/start.S ${BOOT_DIR}/syscalls.c slice.o ${DESIGN_HART_BUILD_magia}/gemm.o ${DESIGN_HART_BUILD_magia}/mesh.o -lm -o ${SLICE_ELF}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/slice.cpp
          ${OUTPUT_DIR}/slice-x.data
          ${OUTPUT_DIR}/slice-w.data
          ${OUTPUT_DIR}/slice-z.data
          ${BOOT_DIR}/start.S
          ${BOOT_DIR}/syscalls.c
          ${DESIGN_DEVICE_STACK}/device.ld
          ${DESIGN_HART_BUILD_magia}/gemm.o
          ${DESIGN_HART_BUILD_magia}/mesh.o
          ${DESIGN_PB}
  COMMENT "Building the bert-base-uncased-emotion slice for the MAGIA device"
  COMMAND_EXPAND_LISTS
  VERBATIM
)
add_custom_target(bert-base-uncased-emotion-slice-build DEPENDS ${SLICE_ELF})

# The GVN slices of this model, which `bbdev gvn-bemu --debug` builds.
add_subdirectory(gvn)
