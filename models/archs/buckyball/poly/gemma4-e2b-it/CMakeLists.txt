set(DESIGN_MODULE examples.models.poly.gemma4-e2b-it)
set(TRACE_SUPPORTED TRUE PARENT_SCOPE)
set(DESIGN_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(MODEL_SOURCE "${REPO_ROOT}/stack/models/models/gemma4-e2b-it")
set(LLVM_OPT_LEVEL 2)
set(LLVM_PASSES "")
set(MATTR "+xbuckyball,+m,+D")
set(PREFILL_LENGTH 16)
set(TRACE_ARGS)
set(TRACE_MODES)
set(TRACE_PASS)
if(CTRACE OR DTRACE)
  list(APPEND TRACE_ARGS --trace --trace-config ${CMAKE_CURRENT_SOURCE_DIR}/trace/trace.toml)
endif()
if(CTRACE)
  list(APPEND TRACE_MODES cycle-trace)
endif()
if(DTRACE)
  list(APPEND TRACE_MODES tensor-trace)
endif()
if(TRACE_MODES)
  list(JOIN TRACE_MODES "," TRACE_MODES)
  set(TRACE_PASS "-convert-trace-to-llvm=${TRACE_MODES}")
endif()
file(GLOB_RECURSE DESIGN_DEPENDS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/trace/*"
  "${CMAKE_CURRENT_SOURCE_DIR}/quant/*.py" "${CMAKE_CURRENT_SOURCE_DIR}/permute/*.py")
list(APPEND DESIGN_DEPENDS
  "${DESIGN_DIR}/kernels.py"
  "${DESIGN_DIR}/attention.py"
  "${DESIGN_DIR}/attention_mxfp8.py"
  "${DESIGN_DIR}/linear.py"
  "${DESIGN_DIR}/partition.py"
  "${DESIGN_DIR}/tasks.py"
  "${DESIGN_DIR}/entry.py"
  "${MODEL_SOURCE}/task_export.py"
  "${MODEL_SOURCE}/task_abi.py"
  "${MODEL_SOURCE}/task_plan.py"
  "${REPO_ROOT}/stack/compiler/quant/mxfp8_graph.py"
  "${REPO_ROOT}/stack/compiler/quant/mxfp8_embedding.py"
  "${REPO_ROOT}/stack/compiler/quant/mxfp8_lowering.py"
  "${REPO_ROOT}/examples/balls/mxmm/compiler/python/layout.py")
set(KERNEL_PASSES
  -eliminate-empty-tensors -convert-elementwise-to-linalg
  -convert-tensor-to-linalg
  -fold-unit-transpose -canonicalize
  -outline-rvv-kernels -fuse-pointwise -outline-rvv-kernels -outline-rvv-matmul
  -canonicalize -cse
  -one-shot-bufferize=bufferize-function-boundaries -lower-rvv-kernels
  -lower-rvv-pointwise
  -lower-gather-rows
  -buffer-deallocation-pipeline=private-function-dynamic-ownership=false
  -optimize-allocation-liveness
  -convert-linalg-to-tile=quantized-only -convert-tile-to-buckyball
  -lower-buckyball-to-bank-ssa -assign-physical-banks
  -llvm-request-c-wrappers -lower-bank-ssa-to-intrinsics
  -expand-strided-metadata -convert-linalg-to-loops
  -lower-buckyball -convert-vector-to-scf=target-rank=0
  -lower-affine -canonicalize -cse
  -convert-scf-to-cf -convert-cf-to-llvm -convert-vector-to-llvm
  -buffer-deallocation-simplification
  -bufferization-lower-deallocations -convert-math-to-llvm
  -convert-math-to-libm -convert-arith-to-llvm -convert-func-to-llvm
  -finalize-memref-to-llvm -reconcile-unrealized-casts -canonicalize -cse
  -lower-tile-runtime -lower-ant-runtime -verify-npu-compute)
string(JOIN " " GEMMA_KERNEL_PIPELINE ${KERNEL_PASSES})
add_subdirectory(${MODEL_SOURCE} prepare)
set(TASK_RUNTIME_DIR "${CMAKE_CURRENT_BINARY_DIR}/task-runtime")
set(TASK_ARCHIVE "${CMAKE_CURRENT_BINARY_DIR}/task-kernels/libgemma4_kernels.a")
add_custom_command(OUTPUT "${TASK_ARCHIVE}"
  COMMAND ${CMAKE_COMMAND} -S "${GENERATED_DIR}/tasks" -B "${CMAKE_CURRENT_BINARY_DIR}/task-kernels"
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_OBJCOPY=${CMAKE_OBJCOPY}
    -DCOMPILER_BUILD=${COMPILER_BUILD} -DGEMMA_RENAMES=${TASK_RUNTIME_DIR}
    -DMODEL_SOURCE=${MODEL_SOURCE} -DREPO_ROOT=${REPO_ROOT}
    -DLLVM_MLIR_EXECUTION_ENGINE_DIR=${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine
  COMMAND ${CMAKE_COMMAND} --build "${CMAKE_CURRENT_BINARY_DIR}/task-kernels"
  DEPENDS "${GENERATED_DIR}/tasks/CMakeLists.txt" "${GENERATED_DIR}/tasks/manifest.json"
          "${TASK_RUNTIME_DIR}/ffn.rename" "${TASK_RUNTIME_DIR}/attention.rename"
  VERBATIM)
add_custom_target(gemma-task-kernels DEPENDS "${TASK_ARCHIVE}")
add_library(gemma4_kernels STATIC IMPORTED)
set_target_properties(gemma4_kernels PROPERTIES IMPORTED_LOCATION "${TASK_ARCHIVE}")
add_dependencies(gemma4_kernels gemma-task-kernels)
find_program(GEMMA_HOST_CXX NAMES c++ REQUIRED NO_CMAKE_FIND_ROOT_PATH)
set(PREPARE_TOOL "${CMAKE_CURRENT_BINARY_DIR}/prepare-input")
add_custom_command(OUTPUT "${PREPARE_TOOL}"
  COMMAND "${GEMMA_HOST_CXX}" -std=c++17 -O2
    -I${BUDDY_SOURCE}/frontend/Interfaces ${MODEL_SOURCE}/prepare.cpp -o "${PREPARE_TOOL}"
  DEPENDS ${MODEL_SOURCE}/prepare.cpp ${BUDDY_SOURCE}/frontend/Interfaces/buddy/LLM/TextContainer.h
  VERBATIM)
add_custom_target(gemma-input-prepare DEPENDS "${PREPARE_TOOL}")
add_executable(gemma4-e2b-it-run ${MODEL_SOURCE}/main.cpp ${MODEL_SOURCE}/generation.cpp ${MODEL_SOURCE}/execution.cpp ${MODEL_SOURCE}/task_worker.cpp ${MODEL_SOURCE}/layer_execution.cpp)
add_dependencies(gemma4-e2b-it-run model-loader gemma-input-prepare gemma4-model-build gemma-task-kernels gemma-target-rvv)
target_include_directories(gemma4-e2b-it-run PRIVATE ${MODEL_SOURCE} ${ISA_DIR}/chip ${BUDDY_SOURCE}/frontend/Interfaces ${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine ${GENERATED_DIR} ${GENERATED_DIR}/tasks)
add_subdirectory("${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/rvv" rvv)
add_subdirectory("${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/mxfp8" mxfp8)
add_subdirectory("${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/mxquant" mxquant)
find_program(GEMMA_TARGET_NM NAMES riscv64-unknown-linux-gnu-nm REQUIRED)
add_custom_command(OUTPUT "${TASK_RUNTIME_DIR}/ffn.rename" "${TASK_RUNTIME_DIR}/attention.rename"
  "${TASK_RUNTIME_DIR}/libgemma_rvv_ffn.a" "${TASK_RUNTIME_DIR}/libgemma_rvv_attention.a"
  COMMAND ${PYTHON} -m stack.models.models.gemma4-e2b-it.task_runtime
    --repo "${REPO_ROOT}" --output "${TASK_RUNTIME_DIR}" --isa "${ISA_DIR}"
    --images "${CMAKE_CURRENT_BINARY_DIR}/rvv" --compiler "${CMAKE_CXX_COMPILER}"
    --ar "${CMAKE_AR}" --nm "${GEMMA_TARGET_NM}" --objcopy "${CMAKE_OBJCOPY}"
  DEPENDS rvv-images ${MODEL_SOURCE}/task_runtime.py
    ${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/rvv/dispatch.cpp
    ${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/rvv/matmul_dispatch.cpp
    ${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/rvv/cache_dispatch.cpp
    ${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/rvv/normalization_dispatch.cpp
    ${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/rvv/pointwise_dispatch.cpp
    ${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/rvv/activation_dispatch.cpp
    ${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/rvv/flash_attention_dispatch.cpp
    ${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/rvv/flash_attention_mxfp8_dispatch.cpp
  WORKING_DIRECTORY "${REPO_ROOT}" VERBATIM)
add_custom_target(gemma-target-rvv DEPENDS "${TASK_RUNTIME_DIR}/ffn.rename" "${TASK_RUNTIME_DIR}/attention.rename"
  "${TASK_RUNTIME_DIR}/libgemma_rvv_ffn.a" "${TASK_RUNTIME_DIR}/libgemma_rvv_attention.a")
target_link_libraries(gemma4-e2b-it-run PRIVATE -static gemma4_kernels
  "${TASK_RUNTIME_DIR}/libgemma_rvv_ffn.a" "${TASK_RUNTIME_DIR}/libgemma_rvv_attention.a"
  runtime_ant_tasks rvv_images mxfp8_pack_ant mxquant_backend_ant)
add_custom_command(TARGET gemma4-e2b-it-run POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${ARTIFACT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:gemma4-e2b-it-run> ${ARTIFACT_DIR}/gemma4-e2b-it-run
  COMMAND ${PYTHON} -m stack.models.models.gemma4-e2b-it.pack
          --generated-dir ${GENERATED_DIR} --model-dir ${ARTIFACT_DIR}
          --chip ${CHIP} --run-config ${DESIGN_DIR}/configs/running-param.toml --tokenizer ${PREPARE_TOOL} --rax-pack ${COMPILER_BUILD}/bin/rax-pack
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
set_property(TARGET gemma4-e2b-it-run APPEND PROPERTY LINK_DEPENDS
  ${MODEL_SOURCE}/pack.py ${MODEL_SOURCE}/inputs.py ${DESIGN_DIR}/configs/running-param.toml ${PREPARE_TOOL} ${REPO_ROOT}/stack/compiler/package.py)
