file(GLOB_RECURSE RECIPE_FRONTEND_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/frontend/*.py")

# What the recipe compiles for this device config; the calibration of the
# model belongs to the recipe.
set(OUTPUT_DIR "${DESIGN_OUTPUT}/models/laya")
set(CALIBRATION "${CMAKE_CURRENT_SOURCE_DIR}/output/calibration.json")
set(WEIGHTS "${MODEL_DIR}/laya/output/laya/typed-decisions")
# The graph is compiled with the plugins of the device and the goban hart
# (DESIGN_HART_OPT_goban); its matmul thread work with the plugins its target
# passes through (DESIGN_TARGET_OPT_matmul).
# The operator image of the GP domain: the operators convert-linalg-to-acc leaves
# in the graph and their entry `operator`, linked by the hart's operators.ld.
set(OPERATOR_LINKER "${DESIGN_HART_STACK_goban}/operators.ld")
math(EXPR OPERATOR_TEXT_BYTES "${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_I_BUF_WORDS} * 4")
math(EXPR MODEL_OFFSET "${DESIGN_DEVICE_DDR_BYTES} - ${DESIGN_DEVICE_DDR_MODEL}")
math(EXPR MODEL_IMAGE "${DESIGN_DEVICE_DDR_BASE} + ${MODEL_OFFSET}" OUTPUT_FORMAT HEXADECIMAL)

# ===----------------------------------------------------------------------===
# Calibrate the Linear input and attention operand scales on typed-decision
# questions; rejected when the decision accuracy drops more than 1 point.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CALIBRATION}
  COMMAND ${PYTHON} buddy-pytorch-calibrate.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-calibrate.py
          ${CMAKE_CURRENT_SOURCE_DIR}/../exp.py
          ${STACK_ROOT}/compiler/frontend/quant.py
          ${MODEL_DIR}/laya/model.py
          ${MODEL_DIR}/laya/data.py
          ${WEIGHTS}/model.safetensors
          ${WEIGHTS}/rl_agent_config.json
          ${WEIGHTS}/encoder/config.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Calibrating laya activations"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Import the laya graph, permute and quantize every matrix product to INT8.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/01-import.txt
         ${OUTPUT_DIR}/02-permuted.txt
         ${OUTPUT_DIR}/03-quanted.txt
         ${OUTPUT_DIR}/arg0.data
         ${OUTPUT_DIR}/arg1.data
         ${OUTPUT_DIR}/arg2.data
         ${OUTPUT_DIR}/forward.mlir
         ${OUTPUT_DIR}/subgraph0.mlir
  COMMAND ${PYTHON} buddy-pytorch-importer.py ${OUTPUT_DIR}
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          ${RECIPE_FRONTEND_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${CMAKE_CURRENT_SOURCE_DIR}/frontend/permute/graph.py
          ${MODEL_DIR}/laya/model.py
          ${MODEL_DIR}/laya/data.py
          ${WEIGHTS}/model.safetensors
          ${WEIGHTS}/rl_agent_config.json
          ${WEIGHTS}/encoder/config.json
          ${CALIBRATION}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing the laya graph"
  VERBATIM
)
add_custom_target(
  laya-import
  DEPENDS ${OUTPUT_DIR}/01-import.txt ${OUTPUT_DIR}/02-permuted.txt ${OUTPUT_DIR}/03-quanted.txt
)

# ===----------------------------------------------------------------------===
# The golden data of the baremetal runner: one held-out typed decision, the
# scores in PyTorch of the smoothed model with the calibrated INT8 operands
# on it and its choices.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/input.data
         ${OUTPUT_DIR}/expected.data
         ${OUTPUT_DIR}/reference.json
         ${OUTPUT_DIR}/dataset/inputs.data
         ${OUTPUT_DIR}/dataset/reference.data
         ${OUTPUT_DIR}/dataset/labels.data
  COMMAND ${PYTHON} buddy-pytorch-golden.py ${OUTPUT_DIR} ${TOOLCHAIN}/lib/libnewlib-reference.so
  DEPENDS ${STACK_ROOT}/compiler/frontend/quant.py
          ${STACK_ROOT}/scripts/nix/riscv.nix
          ${CMAKE_CURRENT_SOURCE_DIR}/../exp.py
          ${CMAKE_CURRENT_SOURCE_DIR}/frontend/quant/graph.py
          ${CMAKE_CURRENT_SOURCE_DIR}/frontend/reference.py
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-golden.py
          ${MODEL_DIR}/laya/model.py
          ${MODEL_DIR}/laya/data.py
          ${CMAKE_SOURCE_DIR}/datasets/typed-decisions/all/train-00000-of-00001.parquet
          ${WEIGHTS}/model.safetensors
          ${WEIGHTS}/rl_agent_config.json
          ${WEIGHTS}/encoder/config.json
          ${CALIBRATION}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Computing the laya golden outputs of a typed decision"
  VERBATIM
)
add_custom_target(laya-golden DEPENDS ${OUTPUT_DIR}/expected.data)
configure_file(laya.rhal.mlir.in ${OUTPUT_DIR}/laya.rhal.mlir @ONLY)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/laya.rax
  COMMAND ${DESIGN_COMPILER}/bin/rax-pack ${OUTPUT_DIR}/laya.rhal.mlir
          --embed-payload
          -o ${OUTPUT_DIR}/laya.rax
  DEPENDS ${OUTPUT_DIR}/laya.rhal.mlir
          ${DESIGN_COMPILER}/bin/rax-pack
          laya-buckyball-goban-model
          laya-buckyball-goban-runner
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)
# Linux runs the RAX entry on the input of the baremetal program and checks it
# against the same reference.
file(WRITE ${OUTPUT_DIR}/run.args "--model\nlaya.rax\n--input\ninput.data\n--reference\nexpected.data\n")
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/model.img
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}/ddr
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${OUTPUT_DIR}/arg0.data ${OUTPUT_DIR}/ddr/arg0.data
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${OUTPUT_DIR}/arg1.data ${OUTPUT_DIR}/ddr/arg1.data
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${OUTPUT_DIR}/arg2.data ${OUTPUT_DIR}/ddr/arg2.data
  COMMAND ${CMAKE_BINARY_DIR}/model-pack ${OUTPUT_DIR}/model.img ${OUTPUT_DIR} laya.rax
            --mapped run.args input.data expected.data
            --mapped ddr/arg0.data
            --mapped ddr/arg1.data
            --mapped ddr/arg2.data
  DEPENDS ${OUTPUT_DIR}/laya.rax
          ${OUTPUT_DIR}/run.args
          ${OUTPUT_DIR}/input.data
          ${OUTPUT_DIR}/expected.data
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
          ${OUTPUT_DIR}/arg2.data
          model-pack-tool
  VERBATIM
)
add_custom_command(
  OUTPUT ${DESIGN_OUTPUT}/workloads/models/goban-laya-baremetal-ddr.bin
  COMMAND ${CMAKE_COMMAND} -E make_directory ${DESIGN_OUTPUT}/workloads/models
  COMMAND ${CMAKE_COMMAND} -E rm -f ${DESIGN_OUTPUT}/workloads/models/goban-laya-baremetal-ddr.bin
  COMMAND dd if=${OUTPUT_DIR}/model.img of=${DESIGN_OUTPUT}/workloads/models/goban-laya-baremetal-ddr.bin bs=1M oflag=seek_bytes seek=${MODEL_OFFSET} conv=sparse status=none
  DEPENDS ${OUTPUT_DIR}/model.img
  VERBATIM
)
add_custom_target(
  laya-model-build
  DEPENDS
    ${DESIGN_OUTPUT}/workloads/models/goban-laya-baremetal
    ${DESIGN_OUTPUT}/workloads/models/goban-laya-baremetal-ddr.bin
    ${OUTPUT_DIR}/model.img
    ${DESIGN_OUTPUT}/workloads/models/goban-laya-dataset
    ${OUTPUT_DIR}/dataset/inputs.data
    ${OUTPUT_DIR}/dataset/reference.data
    ${OUTPUT_DIR}/dataset/labels.data
)
# ===----------------------------------------------------------------------===
# Lower the model entry function to a RISC-V object.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward.o
  BYPRODUCTS ${OUTPUT_DIR}/forward.ll
             ${OUTPUT_DIR}/forward_opt.ll
             ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUTPUT_DIR}/forward.mlir
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
            -finalize-memref-to-llvm=use-aligned-alloc
            -convert-func-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/forward_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/forward.ll
  COMMAND ${LLVM_OPT} ${OUTPUT_DIR}/forward.ll -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=default<O2>" -S -o ${OUTPUT_DIR}/forward_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/forward_opt.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o forward.o
  DEPENDS ${OUTPUT_DIR}/forward.mlir
          ${DESIGN_TARGET_PLUGIN_matmul}
          ${DESIGN_PB}
          ${DESIGN_COMPILER}/bin/buddy-translate
          ${DESIGN_COMPILER}/bin/buddy-llc
  COMMENT "Building forward.o"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower the compute graph and export each compiler stage. The hart pass
# compiles the fused FP32 row regions into operators of the GP domain and
# outlines thread subgraphs; the acc pass maps their INT8 products,
# quantization, bias decoding and row operators to Balls and GP launches. The
# operators leave the host graph (symbol-dce) and become the operator image below. After bufferization, bank assignment
# and lower-acc produce instructions, which lower-hart records as asynchronous
# kernel jobs. Host packing finishes before submission and staging allocations
# survive until completion. Request the graph's C interface before lower-hart
# so kernel entry points keep their C names.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0.ll
             ${OUTPUT_DIR}/subgraph0_opt.ll
             ${OUTPUT_DIR}/subgraph0_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_fused.mlir
             ${OUTPUT_DIR}/subgraph0_host.mlir
             ${OUTPUT_DIR}/subgraph0_hart.mlir
             ${OUTPUT_DIR}/subgraph0_acc.mlir
             ${OUTPUT_DIR}/subgraph0_addresses.mlir
             ${OUTPUT_DIR}/subgraph0_copies.mlir
             ${OUTPUT_DIR}/subgraph0_frontend.mlir
             ${OUTPUT_DIR}/subgraph0_bufferized.mlir
             ${OUTPUT_DIR}/subgraph0_memref.mlir
             ${OUTPUT_DIR}/subgraph0_cwrappers.mlir
             ${OUTPUT_DIR}/subgraph0_physical.mlir
             ${OUTPUT_DIR}/subgraph0_lowered.mlir
             ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUTPUT_DIR}/subgraph0.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" -o ${OUTPUT_DIR}/subgraph0_linalg.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUTPUT_DIR}/subgraph0_linalg.mlir -convert-elementwise-to-linalg -o ${OUTPUT_DIR}/subgraph0_fused.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUTPUT_DIR}/subgraph0_fused.mlir -convert-linalg-to-hart=design=${DESIGN_PB} -o ${OUTPUT_DIR}/subgraph0_hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUTPUT_DIR}/subgraph0_hart.mlir
            -convert-linalg-to-acc
            -o ${OUTPUT_DIR}/subgraph0_acc.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUTPUT_DIR}/subgraph0_acc.mlir "-symbol-privatize=exclude=subgraph0,_binary_operators_image_start" -symbol-dce -o ${OUTPUT_DIR}/subgraph0_host.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUTPUT_DIR}/subgraph0_host.mlir -lower-frontend-fake-op -o ${OUTPUT_DIR}/subgraph0_frontend.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUTPUT_DIR}/subgraph0_frontend.mlir -eliminate-empty-tensors -convert-elementwise-to-linalg -convert-tensor-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -drop-equivalent-buffer-results -buffer-results-to-out-params=hoist-static-allocs -ownership-based-buffer-deallocation -buffer-deallocation-simplification -bufferization-lower-deallocations -canonicalize -o ${OUTPUT_DIR}/subgraph0_bufferized.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUTPUT_DIR}/subgraph0_bufferized.mlir -llvm-request-c-wrappers -o ${OUTPUT_DIR}/subgraph0_cwrappers.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUTPUT_DIR}/subgraph0_cwrappers.mlir
            -assign-physical-banks
            -o ${OUTPUT_DIR}/subgraph0_physical.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUTPUT_DIR}/subgraph0_physical.mlir
            -lower-acc
            -o ${OUTPUT_DIR}/subgraph0_lowered.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUTPUT_DIR}/subgraph0_lowered.mlir -expand-strided-metadata -lower-affine -canonicalize -o ${OUTPUT_DIR}/subgraph0_addresses.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUTPUT_DIR}/subgraph0_addresses.mlir "-lower-hart=design=${DESIGN_PB} hart=goban" -o ${OUTPUT_DIR}/subgraph0_memref.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUTPUT_DIR}/subgraph0_memref.mlir -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUTPUT_DIR}/subgraph0_copies.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUTPUT_DIR}/subgraph0_copies.mlir
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
            -finalize-memref-to-llvm=use-aligned-alloc
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/subgraph0.ll
  COMMAND ${LLVM_OPT} ${OUTPUT_DIR}/subgraph0.ll -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=default<O2>" -S -o ${OUTPUT_DIR}/subgraph0_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0_opt.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -max-store-memcpy=16 -mattr=+xbuckyball,+m,+D -float-abi=hard -o subgraph0.o
  DEPENDS ${OUTPUT_DIR}/subgraph0.mlir
          ${DESIGN_HART_PLUGIN_goban}
          ${DESIGN_PB}
          ${DESIGN_TARGET_PLUGIN_matmul}
          ${DESIGN_COMPILER}/bin/buddy-translate
          ${DESIGN_COMPILER}/bin/buddy-llc
  COMMENT "Building subgraph0.o"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Build the operator image of the GP domain from the operators in the graph and
# embed it as _binary_operators_image_start.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT operators.embed.o
  BYPRODUCTS ${OUTPUT_DIR}/operators.mlir
             ${OUTPUT_DIR}/operators_llvm.mlir
             ${OUTPUT_DIR}/operators.ll
             ${OUTPUT_DIR}/operators_opt.ll
  COMMAND ${DESIGN_HART_OPT_goban} ${OUTPUT_DIR}/subgraph0_acc.mlir "-symbol-privatize=exclude=operator,gp_space" -symbol-dce -o ${OUTPUT_DIR}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${OUTPUT_DIR}/operators.mlir
            -convert-vector-to-scf
            -convert-vector-to-llvm
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-math-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm=use-aligned-alloc
            -convert-func-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/operators_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/operators_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/operators.ll
  COMMAND ${LLVM_OPT} ${OUTPUT_DIR}/operators.ll -mtriple=riscv64 -mattr=+m,+f,+d,+zve64f,+zvl128b "-passes=default<O2>" -S -o ${OUTPUT_DIR}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/operators_opt.ll -filetype=obj -mtriple=riscv64 -O2 -mattr=+m,+f,+d,+zve64f,+zvl128b -riscv-v-vector-bits-min=${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_V_LEN} -riscv-v-vector-bits-max=${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_V_LEN} -target-abi=lp64f -o operators.o
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-ld -m elf64lriscv
            --defsym=operator_text_bytes=${OPERATOR_TEXT_BYTES} -T ${OPERATOR_LINKER} operators.o -o operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata operators.elf operators.image
  COMMAND ${ELF_OBJCOPY} -I binary -O elf64-littleriscv -B riscv
            --rename-section .data=.rodata,alloc,load,readonly,data,contents
            --set-section-alignment .data=16 operators.image operators.embed.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_acc.mlir
          ${OPERATOR_LINKER}
          ${DESIGN_HART_PLUGIN_goban}
          ${LLVM_OPT}
          ${DESIGN_COMPILER}/bin/buddy-opt
          ${DESIGN_COMPILER}/bin/buddy-translate
          ${DESIGN_COMPILER}/bin/buddy-llc
  COMMENT "Building the GP operator image"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Link the model and runtime into a RISC-V shared library.
# ===----------------------------------------------------------------------===
add_library(laya-buckyball-goban-model SHARED EXCLUDE_FROM_ALL entry.cpp forward.o subgraph0.o operators.embed.o)
target_include_directories(laya-buckyball-goban-model
  PRIVATE ${LIB_DIR}
)
target_compile_definitions(laya-buckyball-goban-model
  PRIVATE DEVICE_HARTS_0_SIGNATURE=${DESIGN_DEVICE_HARTS_0_SIGNATURE}ULL
)
set_target_properties(laya-buckyball-goban-model
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             LINKER_LANGUAGE CXX
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN;/lib"
)
target_link_libraries(laya-buckyball-goban-model
  PRIVATE runtime-linux-shared
)

# ===----------------------------------------------------------------------===
# Build the Buddy InferenceRunner plugin for RISC-V Linux.
# ===----------------------------------------------------------------------===
get_filename_component(FLATBUFFERS_HEADER "${TOOLCHAIN}/include/flatbuffers/flatbuffers.h" REALPATH)
get_filename_component(FLATBUFFERS_INCLUDE "${FLATBUFFERS_HEADER}/../.." ABSOLUTE)
add_library(laya-buckyball-goban-runner SHARED EXCLUDE_FROM_ALL runner.cpp)
set_target_properties(laya-buckyball-goban-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN;/lib"
)
target_include_directories(laya-buckyball-goban-runner
  PRIVATE ${LIB_DIR}
          ${RUNTIME_DIR}/services
          ${FLATBUFFERS_INCLUDE}
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/runtime/include
          ${DESIGN_COMPILER}/runtime/include
)
target_link_libraries(laya-buckyball-goban-runner
  PRIVATE dl
)
set(BOOT_DIR "${STACK_ROOT}/runtime/boot")
list(TRANSFORM DESIGN_DEFS PREPEND "-D" OUTPUT_VARIABLE DESIGN_DEFINES)
list(
  APPEND DESIGN_DEFINES
  "-DDEVICE_TRACE_BASE=${DESIGN_DEVICE_TRACE_BASE}ULL"
  "-DDEVICE_TRACE_STRIDE=${DESIGN_DEVICE_TRACE_STRIDE}ULL"
)
list(TRANSFORM DESIGN_TARGET_DEFS_matmul PREPEND "-D" OUTPUT_VARIABLE TARGET_DEFINES)
set(BAREMETAL_ELF "${DESIGN_OUTPUT}/workloads/models/goban-laya-baremetal")
add_custom_command(
  OUTPUT ${BAREMETAL_ELF}
  BYPRODUCTS baremetal.o
             mlir-baremetal.o
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DESIGN_OUTPUT}/workloads/models"
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++20 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -fno-builtin ${DESIGN_DEFINES} -I${LIB_DIR} -I${RUNTIME_DIR}/scheduler -DMODEL_IMAGE=${MODEL_IMAGE}ULL -c ${CMAKE_CURRENT_SOURCE_DIR}/baremetal.cpp -o baremetal.o
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++17 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -fno-builtin -I${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine -c ${RUNTIME_DIR}/services/mlir.cpp -o mlir-baremetal.o
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -specs=nano.specs -specs=nosys.specs -nostartfiles -Wl,-T,${BOOT_DIR}/device.ld ${DESIGN_DEFINES} ${TARGET_DEFINES} -I${RUNTIME_DIR}/services -I${STACK_ROOT}/kernel/include ${BOOT_DIR}/start.S ${BOOT_DIR}/syscalls.c ${STACK_ROOT}/kernel/src/location.c ${RUNTIME_DIR}/trace/baremetal.c baremetal.o mlir-baremetal.o forward.o subgraph0.o operators.embed.o ${DESIGN_HART_BUILD_goban}/execute.o ${DESIGN_THREAD_BUILD_matmul}/loader.o $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:runtime> $<TARGET_FILE:kernel> -lm -lgcc -o ${BAREMETAL_ELF}
  DEPENDS baremetal.cpp
          model.h
          ${LIB_DIR}/MemRef.h
          ${LIB_DIR}/testutils.h
          ${LIB_DIR}/ModelImage.h
          forward.o
          subgraph0.o
          operators.embed.o
          ${BOOT_DIR}/start.S
          ${BOOT_DIR}/syscalls.c
          ${BOOT_DIR}/device.ld
          ${RUNTIME_DIR}/services/mlir.cpp
          ${RUNTIME_DIR}/trace/baremetal.c
          ${RUNTIME_DIR}/trace/event.h
          ${STACK_ROOT}/kernel/src/location.c
          ${STACK_ROOT}/kernel/include/stream.h
          ${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine/CRunnerUtils.h
          ${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine/Msan.h
          ${DESIGN_HART_BUILD_goban}/execute.o
          ${DESIGN_THREAD_BUILD_matmul}/loader.o
          runtime-baremetal
          runtime
          kernel
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
          ${OUTPUT_DIR}/arg2.data
          ${OUTPUT_DIR}/input.data
          ${DESIGN_PB}
  COMMAND_EXPAND_LISTS
  VERBATIM
)

# The accuracy run (dataset.cpp): the decisions of the model region, a shard
# of dataset/ that bench-device --bemu-accuracy packs beside the weights into
# each run's DDR image.
set(DATASET_ELF "${DESIGN_OUTPUT}/workloads/models/goban-laya-dataset")
add_custom_command(
  OUTPUT ${DATASET_ELF}
  BYPRODUCTS dataset.o
             mlir-dataset.o
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DESIGN_OUTPUT}/workloads/models"
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++20 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -fno-builtin ${DESIGN_DEFINES} -I${LIB_DIR} -I${RUNTIME_DIR}/scheduler -DMODEL_IMAGE=${MODEL_IMAGE}ULL -c ${CMAKE_CURRENT_SOURCE_DIR}/dataset.cpp -o dataset.o
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++17 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -fno-builtin -I${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine -c ${RUNTIME_DIR}/services/mlir.cpp -o mlir-dataset.o
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -specs=nano.specs -specs=nosys.specs -nostartfiles -Wl,-T,${BOOT_DIR}/device.ld ${DESIGN_DEFINES} ${TARGET_DEFINES} -I${RUNTIME_DIR}/services -I${STACK_ROOT}/kernel/include ${BOOT_DIR}/start.S ${BOOT_DIR}/syscalls.c ${STACK_ROOT}/kernel/src/location.c ${RUNTIME_DIR}/trace/baremetal.c dataset.o mlir-dataset.o forward.o subgraph0.o operators.embed.o ${DESIGN_HART_BUILD_goban}/execute.o ${DESIGN_THREAD_BUILD_matmul}/loader.o $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:runtime> $<TARGET_FILE:kernel> -lm -lgcc -o ${DATASET_ELF}
  DEPENDS dataset.cpp
          model.h
          ${LIB_DIR}/MemRef.h
          ${LIB_DIR}/testutils.h
          ${LIB_DIR}/ModelImage.h
          forward.o
          subgraph0.o
          operators.embed.o
          ${BOOT_DIR}/start.S
          ${BOOT_DIR}/syscalls.c
          ${BOOT_DIR}/device.ld
          ${RUNTIME_DIR}/services/mlir.cpp
          ${RUNTIME_DIR}/trace/baremetal.c
          ${RUNTIME_DIR}/trace/event.h
          ${STACK_ROOT}/kernel/src/location.c
          ${STACK_ROOT}/kernel/include/stream.h
          ${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine/CRunnerUtils.h
          ${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine/Msan.h
          ${DESIGN_HART_BUILD_goban}/execute.o
          ${DESIGN_THREAD_BUILD_matmul}/loader.o
          runtime-baremetal
          runtime
          kernel
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
          ${OUTPUT_DIR}/arg2.data
          ${DESIGN_PB}
  COMMAND_EXPAND_LISTS
  VERBATIM
)

# The GVN slices of this model, which `bbdev gvn-bemu --debug` builds.
add_subdirectory(gvn)
