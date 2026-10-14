set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

file(GLOB_RECURSE DESIGN_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/trace/*"
  "${CMAKE_CURRENT_SOURCE_DIR}/quant/*.py" "${CMAKE_CURRENT_SOURCE_DIR}/permute/*.py"
)
set(
  MODEL_OUTPUTS
  ${OUTPUT_DIR}/tasks/manifest.json
  ${OUTPUT_DIR}/tasks/task_api.h
  ${OUTPUT_DIR}/tasks/task_index.cpp
  ${OUTPUT_DIR}/tasks/CMakeLists.txt
)
add_custom_command(
  OUTPUT ${MODEL_OUTPUTS}
         ${OUTPUT_DIR}/model.json
         ${OUTPUT_DIR}/gemma-parameters.h
         ${OUTPUT_DIR}/vocab.txt
         ${OUTPUT_DIR}/prefill/params.f32
         ${OUTPUT_DIR}/prefill/weights.bin
         ${OUTPUT_DIR}/decode/params.f32
         ${OUTPUT_DIR}/decode/weights.bin
  COMMAND ${PYTHON} buddy-pytorch-importer.py
            --repo-root ${REPO_ROOT}
            --prefill-length "16"
            --run-config "${CMAKE_CURRENT_SOURCE_DIR}/configs/running-param.toml"
            --isa-dir "${ISA_DIR}"
            --kernel-passes "-eliminate-empty-tensors \
    -convert-elementwise-to-linalg \
    -convert-tensor-to-linalg \
    -fold-unit-transpose \
    -canonicalize \
    -outline-rvv-kernels \
    -fuse-pointwise \
    -outline-rvv-kernels \
    -outline-rvv-matmul \
    -canonicalize \
    -cse \
    -one-shot-bufferize=bufferize-function-boundaries \
    -lower-rvv-kernels \
    -lower-rvv-pointwise \
    -lower-gather-rows \
    -buffer-deallocation-pipeline=private-function-dynamic-ownership=false \
    -optimize-allocation-liveness \
    -convert-linalg-to-tile=quantized-only \
    -convert-tile-to-buckyball \
    -lower-buckyball-to-bank-ssa \
    -assign-physical-banks \
    -llvm-request-c-wrappers \
    -lower-bank-ssa-to-intrinsics \
    -expand-strided-metadata \
    -convert-linalg-to-loops \
    -lower-buckyball \
    -convert-vector-to-scf=target-rank=0 \
    -lower-affine \
    -canonicalize \
    -cse \
    -convert-scf-to-cf \
    -convert-cf-to-llvm \
    -convert-vector-to-llvm \
    -buffer-deallocation-simplification \
    -bufferization-lower-deallocations \
    -convert-math-to-llvm \
    -convert-math-to-libm \
    -convert-arith-to-llvm \
    -convert-func-to-llvm \
    -finalize-memref-to-llvm \
    -reconcile-unrealized-casts \
    -canonicalize \
    -cse \
    -lower-ant-runtime \
    -verify-npu-compute"
            --checkpoint "google/gemma-4-E2B-it"
            --chip ${CHIP}
            --output-dir ${OUTPUT_DIR}
            --compiler-build ${COMPILER_BUILD}
            --chip-pb ${CHIP_PB}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/configs/importer-param.py
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${CMAKE_CURRENT_SOURCE_DIR}/abi.py
          ${CMAKE_CURRENT_SOURCE_DIR}/cache.py
          ${CMAKE_CURRENT_SOURCE_DIR}/native.py
          ${MODEL_DIR}/gemma4-e2b-it/prepare.py
          ${DESIGN_FILES}
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/attention.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/attention_mxfp8.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/linear.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/partition.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/tasks.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/entry.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/task_export.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/task_abi.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/task_plan.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/task_program.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/workspace.py"
          "${REPO_ROOT}/stack/compiler/quant/mxfp8_graph.py"
          "${REPO_ROOT}/stack/compiler/quant/mxfp8_embedding.py"
          "${REPO_ROOT}/stack/compiler/quant/mxfp8_lowering.py"
          ${CMAKE_CURRENT_SOURCE_DIR}/configs/running-param.toml
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)

add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/task-kernels/libgemma4_kernels.a"
  COMMAND ${CMAKE_COMMAND} -S "${OUTPUT_DIR}/tasks" -B "${CMAKE_CURRENT_BINARY_DIR}/task-kernels" -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_OBJCOPY=${CMAKE_OBJCOPY} -DCOMPILER_BUILD=${COMPILER_BUILD} -DGEMMA_RENAMES=${CMAKE_CURRENT_BINARY_DIR}/task-runtime -DDESIGN_DIR=${CMAKE_CURRENT_SOURCE_DIR} -DREPO_ROOT=${REPO_ROOT} -DLLVM_MLIR_EXECUTION_ENGINE_DIR=${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine
  COMMAND ${CMAKE_COMMAND}
            --build "${CMAKE_CURRENT_BINARY_DIR}/task-kernels"
  DEPENDS "${OUTPUT_DIR}/tasks/CMakeLists.txt"
          "${OUTPUT_DIR}/tasks/manifest.json"
          "${CMAKE_CURRENT_BINARY_DIR}/task-runtime/ffn.rename"
          "${CMAKE_CURRENT_BINARY_DIR}/task-runtime/attention.rename"
  VERBATIM
)
add_custom_target(gemma-task-kernels DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/task-kernels/libgemma4_kernels.a")
add_library(gemma4-e2b-it-buckyball-poly-model STATIC IMPORTED)
set_target_properties(gemma4-e2b-it-buckyball-poly-model PROPERTIES IMPORTED_LOCATION
  "${CMAKE_CURRENT_BINARY_DIR}/task-kernels/libgemma4_kernels.a"
)
add_dependencies(gemma4-e2b-it-buckyball-poly-model gemma-task-kernels)
find_program(GEMMA_HOST_CXX NAMES c++ REQUIRED NO_CMAKE_FIND_ROOT_PATH)
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/prepare-input"
  COMMAND "${GEMMA_HOST_CXX}" -std=c++17 -O2 -I${BUDDY_SOURCE}/frontend/Interfaces ${MODEL_DIR}/gemma4-e2b-it/prepare.cpp -o "${CMAKE_CURRENT_BINARY_DIR}/prepare-input"
  DEPENDS ${MODEL_DIR}/gemma4-e2b-it/prepare.cpp
          ${BUDDY_SOURCE}/frontend/Interfaces/buddy/LLM/TextContainer.h
  VERBATIM
)
add_custom_target(gemma-input-prepare DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/prepare-input")
add_executable(gemma4-e2b-it-run EXCLUDE_FROM_ALL ${MODEL_DIR}/gemma4-e2b-it/main.cpp
  ${MODEL_DIR}/gemma4-e2b-it/generation.cpp ${CMAKE_CURRENT_SOURCE_DIR}/execution.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/loader.cpp ${CMAKE_CURRENT_SOURCE_DIR}/task_worker.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/layer_execution.cpp
)
set_target_properties(gemma4-e2b-it-run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
target_compile_features(gemma4-e2b-it-run PRIVATE cxx_std_20)
add_dependencies(gemma4-e2b-it-run model-loader gemma-input-prepare gemma-task-kernels gemma-target-rvv)
target_include_directories(gemma4-e2b-it-run PRIVATE ${MODEL_DIR}/gemma4-e2b-it
                                                     ${CMAKE_CURRENT_SOURCE_DIR}
                                                     ${ISA_DIR}/chip
                                                     ${BUDDY_SOURCE}/frontend/Interfaces
                                                     ${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine
                                                     ${OUTPUT_DIR}
                                                     ${OUTPUT_DIR}/tasks
)
find_program(GEMMA_TARGET_NM NAMES riscv64-unknown-linux-gnu-nm REQUIRED)
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/task-runtime/ffn.rename"
         "${CMAKE_CURRENT_BINARY_DIR}/task-runtime/attention.rename"
         "${CMAKE_CURRENT_BINARY_DIR}/task-runtime/libgemma_rvv_ffn.a"
         "${CMAKE_CURRENT_BINARY_DIR}/task-runtime/libgemma_rvv_attention.a"
  COMMAND ${PYTHON} task_runtime.py
            --repo "${REPO_ROOT}"
            --output "${CMAKE_CURRENT_BINARY_DIR}/task-runtime"
            --isa "${ISA_DIR}"
            --images "${RVV_BUILD_DIR}"
            --compiler "${CMAKE_CXX_COMPILER}"
            --thread-runtime "${THREAD_RUNTIME_DIR}"
            --ar "${CMAKE_AR}"
            --nm "${GEMMA_TARGET_NM}"
            --objcopy "${CMAKE_OBJCOPY}"
  DEPENDS rvv-images
          ${CMAKE_CURRENT_SOURCE_DIR}/task_runtime.py
          ${THREAD_RUNTIME_DIR}/ffn.rename
          ${THREAD_RUNTIME_DIR}/attention.rename
          ${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/rvv/dispatch.cpp
          ${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/rvv/matmul_dispatch.cpp
          ${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/rvv/cache_dispatch.cpp
          ${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/rvv/normalization_dispatch.cpp
          ${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/rvv/pointwise_dispatch.cpp
          ${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/rvv/activation_dispatch.cpp
          ${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/rvv/flash_attention_dispatch.cpp
          ${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/rvv/flash_attention_mxfp8_dispatch.cpp
  WORKING_DIRECTORY "${REPO_ROOT}"
  VERBATIM
)
add_custom_target(
  gemma-target-rvv
  DEPENDS
    "${CMAKE_CURRENT_BINARY_DIR}/task-runtime/ffn.rename"
    "${CMAKE_CURRENT_BINARY_DIR}/task-runtime/attention.rename"
    "${CMAKE_CURRENT_BINARY_DIR}/task-runtime/libgemma_rvv_ffn.a"
    "${CMAKE_CURRENT_BINARY_DIR}/task-runtime/libgemma_rvv_attention.a"
)
target_link_libraries(gemma4-e2b-it-run PRIVATE -static
                                                gemma4-e2b-it-buckyball-poly-model
                                                "${CMAKE_CURRENT_BINARY_DIR}/task-runtime/libgemma_rvv_ffn.a"
                                                "${CMAKE_CURRENT_BINARY_DIR}/task-runtime/libgemma_rvv_attention.a"
                                                runtime_ant_tasks
                                                rvv_images
                                                thread_runtime_ant
)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/gemma4-e2b-it.rax
  BYPRODUCTS ${OUTPUT_DIR}/layout.json
             ${OUTPUT_DIR}/gemma4-e2b-it.rhal.mlir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
  COMMAND ${PYTHON} pack.py
            --repo-root ${REPO_ROOT}
            --output-dir ${OUTPUT_DIR}
            --chip ${CHIP}
            --run-config ${CMAKE_CURRENT_SOURCE_DIR}/configs/running-param.toml
            --tokenizer ${CMAKE_CURRENT_BINARY_DIR}/prepare-input
            --rax-pack ${COMPILER_BUILD}/bin/rax-pack
  DEPENDS gemma4-e2b-it-run
          ${CMAKE_CURRENT_SOURCE_DIR}/pack.py
          ${CMAKE_CURRENT_SOURCE_DIR}/inputs.py
          ${CMAKE_CURRENT_SOURCE_DIR}/configs/running-param.toml
          ${CMAKE_CURRENT_BINARY_DIR}/prepare-input
          ${MODEL_OUTPUTS}
          ${OUTPUT_DIR}/model.json
          ${OUTPUT_DIR}/gemma-parameters.h
          ${OUTPUT_DIR}/vocab.txt
          ${OUTPUT_DIR}/prefill/params.f32
          ${OUTPUT_DIR}/prefill/weights.bin
          ${OUTPUT_DIR}/decode/params.f32
          ${OUTPUT_DIR}/decode/weights.bin
          ${REPO_ROOT}/examples/models/poly/gemma4-e2b-it/pack.py
          ${COMPILER_BUILD}/bin/rax-pack
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging gemma4-e2b-it model"
  VERBATIM
)
add_custom_target(gemma4-e2b-it-model-build DEPENDS ${OUTPUT_DIR}/gemma4-e2b-it.rax)
