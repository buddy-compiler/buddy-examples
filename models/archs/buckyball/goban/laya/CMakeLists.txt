file(GLOB_RECURSE RECIPE_FRONTEND_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/frontend/*.py")

set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")
set(WEIGHTS "${MODEL_DIR}/laya/output/laya/typed-decisions")
set(CORE_COMPILER "${DESIGN_CORE_BUILD_goban}/libcore-compiler.so")
set(
  CORE_OPT
  ${DESIGN_COMPILER}/bin/buddy-opt
  --load-dialect-plugin=${CORE_COMPILER}
  --load-pass-plugin=${CORE_COMPILER}
)
set(ACC_OPT ${CORE_OPT} --buckyball-design-pb=${DESIGN_PB} --buckyball-target=matmul)
math(EXPR MODEL_OFFSET "${DESIGN_DEVICE_DDR_BYTES} - ${DESIGN_DEVICE_DDR_MODEL}")
math(EXPR MODEL_IMAGE "${DESIGN_DEVICE_DDR_BASE} + ${MODEL_OFFSET}" OUTPUT_FORMAT HEXADECIMAL)

# ===----------------------------------------------------------------------===
# Calibrate the Linear input and attention operand scales on typed-decision
# questions; rejected when the decision accuracy drops more than 1 point.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/calibration.json
  COMMAND ${PYTHON} buddy-pytorch-calibrate.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-calibrate.py
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
         ${OUTPUT_DIR}/input.data
         ${OUTPUT_DIR}/expected.data
  COMMAND ${PYTHON} buddy-pytorch-importer.py
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          ${RECIPE_FRONTEND_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${CMAKE_CURRENT_SOURCE_DIR}/frontend/permute/graph.py
          ${MODEL_DIR}/laya/model.py
          ${MODEL_DIR}/laya/data.py
          ${WEIGHTS}/model.safetensors
          ${WEIGHTS}/rl_agent_config.json
          ${WEIGHTS}/encoder/config.json
          ${OUTPUT_DIR}/calibration.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing the laya graph"
  VERBATIM
)
add_custom_target(
  laya-import
  DEPENDS ${OUTPUT_DIR}/01-import.txt ${OUTPUT_DIR}/02-permuted.txt ${OUTPUT_DIR}/03-quanted.txt
)
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
file(WRITE ${OUTPUT_DIR}/run.args "--model\nlaya.rax\n--input\ninput.data\n")
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
)
# ===----------------------------------------------------------------------===
# Lower the model entry function to a RISC-V object.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward.o
  BYPRODUCTS ${OUTPUT_DIR}/forward.ll
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
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -expand-strided-metadata
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/forward_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/forward.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/forward.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o forward.o
  DEPENDS ${OUTPUT_DIR}/forward.mlir
          ${DESIGN_TARGET_PLUGIN_matmul}
          ${DESIGN_PB}
          ${DESIGN_COMPILER}/bin/buddy-translate
          ${DESIGN_COMPILER}/bin/buddy-llc
  COMMENT "Building forward.o"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower the compute graph and export each compiler stage. The core pass
# outlines thread subgraphs; the acc pass maps their INT8 products,
# quantization and bias decoding to Balls. After bufferization, bank assignment
# and lower-acc produce instructions, which lower-core records as asynchronous
# engine jobs. Host packing finishes before submission and staging allocations
# survive until completion. Request the graph's C interface before lower-core
# so engine entry points keep their C names.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0.ll
             ${OUTPUT_DIR}/subgraph0_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_core.mlir
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
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/subgraph0.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" -o ${OUTPUT_DIR}/subgraph0_linalg.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/subgraph0_linalg.mlir -convert-linalg-to-core=design=${DESIGN_PB} -o ${OUTPUT_DIR}/subgraph0_core.mlir
  COMMAND ${ACC_OPT} ${OUTPUT_DIR}/subgraph0_core.mlir -convert-linalg-to-acc -o ${OUTPUT_DIR}/subgraph0_acc.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/subgraph0_acc.mlir -lower-frontend-fake-op -o ${OUTPUT_DIR}/subgraph0_frontend.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/subgraph0_frontend.mlir -eliminate-empty-tensors -convert-elementwise-to-linalg -convert-tensor-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -drop-equivalent-buffer-results -buffer-results-to-out-params=hoist-static-allocs -ownership-based-buffer-deallocation -buffer-deallocation-simplification -bufferization-lower-deallocations -canonicalize -o ${OUTPUT_DIR}/subgraph0_bufferized.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/subgraph0_bufferized.mlir -llvm-request-c-wrappers -o ${OUTPUT_DIR}/subgraph0_cwrappers.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUTPUT_DIR}/subgraph0_cwrappers.mlir
            -assign-physical-banks
            -o ${OUTPUT_DIR}/subgraph0_physical.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUTPUT_DIR}/subgraph0_physical.mlir
            -lower-acc
            -o ${OUTPUT_DIR}/subgraph0_lowered.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/subgraph0_lowered.mlir -expand-strided-metadata -lower-affine -canonicalize -o ${OUTPUT_DIR}/subgraph0_addresses.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/subgraph0_addresses.mlir "-lower-core=design=${DESIGN_PB} core=goban" -o ${OUTPUT_DIR}/subgraph0_memref.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/subgraph0_memref.mlir -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUTPUT_DIR}/subgraph0_copies.mlir
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
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/subgraph0.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o subgraph0.o
  DEPENDS ${OUTPUT_DIR}/subgraph0.mlir
          ${CORE_COMPILER}
          ${DESIGN_PB}
          ${DESIGN_TARGET_PLUGIN_matmul}
          ${DESIGN_COMPILER}/bin/buddy-translate
          ${DESIGN_COMPILER}/bin/buddy-llc
  COMMENT "Building subgraph0.o"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Link the model and runtime into a RISC-V shared library.
# ===----------------------------------------------------------------------===
add_library(laya-buckyball-goban-model SHARED EXCLUDE_FROM_ALL entry.cpp forward.o subgraph0.o)
target_include_directories(laya-buckyball-goban-model
  PRIVATE ${LIB_DIR}
)
target_compile_definitions(laya-buckyball-goban-model
  PRIVATE DEVICE_CORES_0_SIGNATURE=${DESIGN_DEVICE_CORES_0_SIGNATURE}ULL
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
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -specs=nano.specs -specs=nosys.specs -nostartfiles -Wl,-T,${BOOT_DIR}/device.ld ${DESIGN_DEFINES} ${TARGET_DEFINES} -I${RUNTIME_DIR}/services -I${STACK_ROOT}/engine/include ${BOOT_DIR}/start.S ${BOOT_DIR}/syscalls.c ${RUNTIME_DIR}/services/dma.c ${RUNTIME_DIR}/services/dma_memory.c ${STACK_ROOT}/engine/src/location.c ${RUNTIME_DIR}/trace/baremetal.c baremetal.o mlir-baremetal.o forward.o subgraph0.o ${DESIGN_CORE_BUILD_goban}/execute.o ${DESIGN_THREAD_BUILD_matmul}/loader.o $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:runtime> $<TARGET_FILE:engine> -lm -lgcc -o ${BAREMETAL_ELF}
  DEPENDS baremetal.cpp
          model.h
          ${LIB_DIR}/MemRef.h
          ${LIB_DIR}/testutils.h
          ${LIB_DIR}/ModelImage.h
          forward.o
          subgraph0.o
          ${BOOT_DIR}/start.S
          ${BOOT_DIR}/syscalls.c
          ${BOOT_DIR}/device.ld
          ${RUNTIME_DIR}/services/dma.c
          ${RUNTIME_DIR}/services/dma.h
          ${RUNTIME_DIR}/services/dma_memory.c
          ${RUNTIME_DIR}/services/dma_memory.h
          ${RUNTIME_DIR}/services/mlir.cpp
          ${RUNTIME_DIR}/trace/baremetal.c
          ${RUNTIME_DIR}/trace/event.h
          ${STACK_ROOT}/engine/src/location.c
          ${STACK_ROOT}/engine/include/stream.h
          ${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine/CRunnerUtils.h
          ${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine/Msan.h
          ${DESIGN_CORE_BUILD_goban}/execute.o
          ${DESIGN_THREAD_BUILD_matmul}/loader.o
          runtime-baremetal
          runtime
          engine
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
          ${OUTPUT_DIR}/arg2.data
          ${OUTPUT_DIR}/input.data
          ${DESIGN_PB}
  COMMAND_EXPAND_LISTS
  VERBATIM
)
