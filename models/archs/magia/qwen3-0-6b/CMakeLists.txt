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
         ${OUTPUT_DIR}/input-order.txt
         ${OUTPUT_DIR}/check-prompt.data
         ${OUTPUT_DIR}/check-tokens.data
         ${OUTPUT_DIR}/check-logits.data
         ${OUTPUT_DIR}/slice-x.data
         ${OUTPUT_DIR}/slice-w.data
         ${OUTPUT_DIR}/slice-z.data
  COMMAND ${PYTHON} buddy-pytorch-importer.py ${OUTPUT_DIR}/libmagia-gemm.so
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          ${OUTPUT_DIR}/libmagia-gemm.so
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${CMAKE_CURRENT_SOURCE_DIR}/frontend/weights.py
          ${CMAKE_CURRENT_SOURCE_DIR}/cache.py
          ${MODEL_DIR}/qwen3-0-6b/model.py
          ${MODEL_DIR}/qwen3-0-6b/input.txt
          ${MODEL_DIR}/qwen3-0-6b/output/qwen3-0-6b/config.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing qwen3-0-6b for magia"
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
  COMMENT "Lowering the qwen3-0-6b entry for magia"
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
            -canonicalize
            -optimize-allocation-liveness
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
  COMMENT "Lowering the qwen3-0-6b graph for magia"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The check of this recipe: the model compiled for this machine, the mesh's
# products in software, its tokens against the model in PyTorch with the
# same products.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/qwen3-0-6b-magia-check
  BYPRODUCTS ${OUTPUT_DIR}/check-forward.o
             ${OUTPUT_DIR}/check-subgraph0.o
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/forward_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/forward.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward.ll -filetype=obj -relocation-model=pic -O2 -o ${OUTPUT_DIR}/check-forward.o
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/subgraph0.ll
  # The address arithmetic of the dynamic sizes is simplified before the
  # code generator's loop passes read it.
  COMMAND ${LLVM_OPT} ${OUTPUT_DIR}/subgraph0.ll "-passes=default<O2>" -S -o ${OUTPUT_DIR}/subgraph0_opt.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0_opt.ll -filetype=obj -relocation-model=pic -O2 -o ${OUTPUT_DIR}/check-subgraph0.o
  COMMAND ${TOOLCHAIN}/bin/g++ -std=c++20 -O2 ${CMAKE_CURRENT_SOURCE_DIR}/check.cpp ${OUTPUT_DIR}/check-forward.o ${OUTPUT_DIR}/check-subgraph0.o ${OUTPUT_DIR}/libmagia-gemm.so -Wl,-rpath,${OUTPUT_DIR} -L${BUDDY_SOURCE}/llvm/build/lib -Wl,-rpath,${BUDDY_SOURCE}/llvm/build/lib -lmlir_c_runner_utils -lmlir_runner_utils -lm -o ${OUTPUT_DIR}/qwen3-0-6b-magia-check
  DEPENDS ${OUTPUT_DIR}/forward_llvm.mlir
          ${OUTPUT_DIR}/subgraph0_llvm.mlir
          ${CMAKE_CURRENT_SOURCE_DIR}/check.cpp
          ${OUTPUT_DIR}/libmagia-gemm.so
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Building the qwen3-0-6b check for magia"
  VERBATIM
)
add_custom_target(
  qwen3-0-6b-model-check
  COMMAND ${OUTPUT_DIR}/qwen3-0-6b-magia-check ${OUTPUT_DIR}
  DEPENDS
    ${OUTPUT_DIR}/qwen3-0-6b-magia-check
    ${OUTPUT_DIR}/arg0.data
    ${OUTPUT_DIR}/arg1.data
    ${OUTPUT_DIR}/input-order.txt
    ${OUTPUT_DIR}/check-prompt.data
    ${OUTPUT_DIR}/check-tokens.data
    ${OUTPUT_DIR}/check-logits.data
  COMMENT "Running qwen3-0-6b for magia with the mesh in software"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Loom runs the compiled graph and launches matrix products on the mesh.
# Its 1.8 GiB of weights reside in the model region above the bare-metal heap.
# ===----------------------------------------------------------------------===
math(EXPR MODEL_OFFSET "${DESIGN_DEVICE_DDR_BYTES} - ${DESIGN_DEVICE_DDR_MODEL}")
math(EXPR MODEL_BASE "${DESIGN_DEVICE_DDR_BASE} + ${MODEL_OFFSET}" OUTPUT_FORMAT HEXADECIMAL)
set(BAREMETAL_ELF "${DESIGN_OUTPUT}/workloads/models/magia-qwen3-0-6b-baremetal")
add_custom_command(
  OUTPUT forward.o
         subgraph0.o
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/forward_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/forward.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c,+zfh -target-abi=lp64d -code-model=medium -O2 -o forward.o
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/subgraph0.ll
  COMMAND ${LLVM_OPT} ${OUTPUT_DIR}/subgraph0.ll -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c,+zfh "-passes=default<O2>" -S -o ${OUTPUT_DIR}/subgraph0_riscv.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0_riscv.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c,+zfh -target-abi=lp64d -code-model=medium -O2 -o subgraph0.o
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
            --embed-dir=${OUTPUT_DIR} -DMODEL_BASE=${MODEL_BASE}ULL -c ${CMAKE_CURRENT_SOURCE_DIR}/baremetal.cpp -o baremetal.o
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++26 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -I${LIB_DIR}
            --embed-dir=${OUTPUT_DIR} -DMODEL_BASE=${MODEL_BASE}ULL -c ${CMAKE_CURRENT_SOURCE_DIR}/checked.cpp -o checked.o
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++17 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -fno-builtin -I${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine -c ${RUNTIME_DIR}/services/mlir.cpp -o mlir-baremetal.o
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -specs=nano.specs -specs=nosys.specs -nostartfiles -Wl,-T,${DESIGN_DEVICE_STACK}/device.ld ${DESIGN_DEFINES} ${BOOT_DIR}/start.S ${BOOT_DIR}/syscalls.c baremetal.o checked.o mlir-baremetal.o forward.o subgraph0.o ${DESIGN_HART_BUILD_magia}/gemm.o ${DESIGN_HART_BUILD_magia}/mesh.o -lstdc++ -lm -o ${BAREMETAL_ELF}
  DEPENDS forward.o
          subgraph0.o
          checked.h
          checked.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/baremetal.cpp
          ${LIB_DIR}/ModelImage.h
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
          ${OUTPUT_DIR}/input-order.txt
          ${OUTPUT_DIR}/check-prompt.data
          ${OUTPUT_DIR}/check-tokens.data
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
  COMMENT "Building qwen3-0-6b for the MAGIA device"
  COMMAND_EXPAND_LISTS
  VERBATIM
)

# The accuracy run (dataset.cpp): LAMBADA passages and the FP32 checkpoint's
# predictions in models/qwen3-0-6b/dataset of the design, the weights under
# its model/, where bench-device --bemu-accuracy cuts the passages into the
# shards of the model region of each run's DDR image.
set(DATASET_DIR "${DESIGN_OUTPUT}/models/qwen3-0-6b")
add_custom_command(
  OUTPUT ${DATASET_DIR}/dataset/contexts.data
         ${DATASET_DIR}/dataset/targets.data
         ${DATASET_DIR}/dataset/lengths.data
         ${DATASET_DIR}/dataset/reference.data
         ${DATASET_DIR}/dataset/labels.data
  COMMAND ${PYTHON} buddy-pytorch-dataset.py ${DATASET_DIR} ${OUTPUT_DIR}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-dataset.py
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
          ${MODEL_DIR}/qwen3-0-6b/model.py
          ${MODEL_DIR}/qwen3-0-6b/output/qwen3-0-6b/config.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Preparing the LAMBADA passages of qwen3-0-6b for magia"
  VERBATIM
)
set(DATASET_ELF "${DESIGN_OUTPUT}/workloads/models/magia-qwen3-0-6b-dataset")
add_custom_command(
  OUTPUT ${DATASET_ELF}
  BYPRODUCTS dataset.o
             mlir-dataset.o
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DESIGN_OUTPUT}/workloads/models"
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++26 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -I${LIB_DIR}
            --embed-dir=${OUTPUT_DIR} -DMODEL_BASE=${MODEL_BASE}ULL -c ${CMAKE_CURRENT_SOURCE_DIR}/dataset.cpp -o dataset.o
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++17 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -fno-builtin -I${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine -c ${RUNTIME_DIR}/services/mlir.cpp -o mlir-dataset.o
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -specs=nano.specs -specs=nosys.specs -nostartfiles -Wl,-T,${DESIGN_DEVICE_STACK}/device.ld ${DESIGN_DEFINES} ${BOOT_DIR}/start.S ${BOOT_DIR}/syscalls.c dataset.o mlir-dataset.o forward.o subgraph0.o ${DESIGN_HART_BUILD_magia}/gemm.o ${DESIGN_HART_BUILD_magia}/mesh.o -lstdc++ -lm -o ${DATASET_ELF}
  DEPENDS forward.o
          subgraph0.o
          ${CMAKE_CURRENT_SOURCE_DIR}/dataset.cpp
          ${LIB_DIR}/ModelImage.h
          ${OUTPUT_DIR}/input-order.txt
          ${BOOT_DIR}/start.S
          ${BOOT_DIR}/syscalls.c
          ${DESIGN_DEVICE_STACK}/device.ld
          ${RUNTIME_DIR}/services/mlir.cpp
          ${DESIGN_HART_BUILD_magia}/gemm.o
          ${DESIGN_HART_BUILD_magia}/mesh.o
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
          ${DESIGN_PB}
  COMMENT "Building the qwen3-0-6b accuracy run for the MAGIA device"
  COMMAND_EXPAND_LISTS
  VERBATIM
)
add_custom_command(
  OUTPUT ${BAREMETAL_ELF}-ddr.bin
  BYPRODUCTS weights.img
  COMMAND ${CMAKE_BINARY_DIR}/model-pack weights.img ${OUTPUT_DIR}
            --mapped arg0.data
            --mapped arg1.data
  COMMAND ${CMAKE_COMMAND} -E rm -f ${BAREMETAL_ELF}-ddr.bin
  COMMAND dd if=weights.img of=${BAREMETAL_ELF}-ddr.bin bs=1M oflag=seek_bytes seek=${MODEL_OFFSET} conv=sparse status=none
  DEPENDS model-pack-tool
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
          ${DESIGN_PB}
          ${BAREMETAL_ELF}
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
add_library(qwen3-0-6b-magia-runner SHARED EXCLUDE_FROM_ALL runner.cpp checked.cpp forward.o subgraph0.o
  ${DESIGN_HART_BUILD_magia}/gemm-linux.o ${DESIGN_HART_BUILD_magia}/firmware-linux.o
)
target_compile_options(qwen3-0-6b-magia-runner PRIVATE --embed-dir=${OUTPUT_DIR})
target_include_directories(qwen3-0-6b-magia-runner
  PRIVATE ${LIB_DIR}
          ${RUNTIME_DIR}/services
          ${FLATBUFFERS_INCLUDE}
          ${BUDDY_SOURCE}/runtime/include
          ${COMPILER_BUILD}/runtime/include
)
target_link_libraries(qwen3-0-6b-magia-runner
  PRIVATE magia-runtime
          m
)
set_target_properties(qwen3-0-6b-magia-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             CXX_STANDARD 26
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN;/lib"
)
configure_file(qwen3-0-6b.rhal.mlir.in ${OUTPUT_DIR}/qwen3-0-6b.rhal.mlir @ONLY)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/qwen3-0-6b.rax
  COMMAND ${COMPILER_BUILD}/bin/rax-pack ${OUTPUT_DIR}/qwen3-0-6b.rhal.mlir
          --embed-payload
          -o ${OUTPUT_DIR}/qwen3-0-6b.rax
  DEPENDS ${OUTPUT_DIR}/qwen3-0-6b.rhal.mlir
          qwen3-0-6b-magia-runner
          ${COMPILER_BUILD}/bin/rax-pack
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)
file(WRITE ${OUTPUT_DIR}/run.args "--model\nqwen3-0-6b.rax\n--input\ncheck-prompt.data\n")
set(MODEL_IMAGE_FILE "${DESIGN_OUTPUT}/models/qwen3-0-6b/model.img")
add_custom_command(
  OUTPUT ${MODEL_IMAGE_FILE}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DESIGN_OUTPUT}/models/qwen3-0-6b"
  COMMAND ${CMAKE_BINARY_DIR}/model-pack ${MODEL_IMAGE_FILE} ${OUTPUT_DIR} qwen3-0-6b.rax check-prompt.data
            --mapped run.args
            --mapped arg0.data
            --mapped arg1.data
  DEPENDS model-pack-tool
          ${OUTPUT_DIR}/qwen3-0-6b.rax
          ${OUTPUT_DIR}/run.args
          ${OUTPUT_DIR}/check-prompt.data
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
  VERBATIM
)
add_custom_target(
  qwen3-0-6b-model-build
  DEPENDS
    ${BAREMETAL_ELF}
    ${BAREMETAL_ELF}-ddr.bin
    ${MODEL_IMAGE_FILE}
    ${DATASET_ELF}
    ${DATASET_DIR}/dataset/labels.data
)

# ===----------------------------------------------------------------------===
# A slice of the model for the RTL of the device (slice.cpp): the first
# product of a decode step, on the mesh, against the reference run's values.
# ===----------------------------------------------------------------------===
set(SLICE_ELF "${DESIGN_OUTPUT}/workloads/models/magia-qwen3-0-6b-slice-baremetal")
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
  COMMENT "Building the qwen3-0-6b slice for the MAGIA device"
  COMMAND_EXPAND_LISTS
  VERBATIM
)
add_custom_target(qwen3-0-6b-slice-build DEPENDS ${SLICE_ELF})

# The GVN slices of this model, which `bbdev gvn-bemu --debug` builds.
add_subdirectory(gvn)
