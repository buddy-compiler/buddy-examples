set(DESIGN_MODULE examples.models.poly.llama3-2-3b-instruct)
set(CHECKPOINT "meta-llama/Llama-3.2-3B-Instruct")
set(DESIGN_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(LLVM_OPT_LEVEL 2 CACHE STRING "Backend optimization level")
set(LLVM_PASSES "function(sroa,early-cse,instcombine,simplifycfg,adce)")
set(MATTR "+xbuckyball,+m,+D")
set(MODEL_SOURCE "${REPO_ROOT}/stack/models/models/llama3-2-3b-instruct")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${DESIGN_DIR}/configs/model.toml")
execute_process(COMMAND ${PYTHON} -c
  "import sys,tomllib; p=tomllib.load(open(sys.argv[1], 'rb'))['parameters']; print(p['prefill_length']); print(p['cache_length'])"
  "${DESIGN_DIR}/configs/model.toml" OUTPUT_VARIABLE DESIGN_CAPACITIES
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE "\n" ";" DESIGN_CAPACITIES "${DESIGN_CAPACITIES}")
list(GET DESIGN_CAPACITIES 0 DESIGN_PREFILL_LENGTH)
list(GET DESIGN_CAPACITIES 1 DESIGN_CACHE_LENGTH)
set(PREFILL_LENGTH ${DESIGN_PREFILL_LENGTH} CACHE STRING "Compiled prefill capacity")
set(CACHE_LENGTH ${DESIGN_CACHE_LENGTH} CACHE STRING "Compiled KV capacity")
file(STRINGS "${ISA_DIR}/chip/topology.h" TILE_COUNT_LINE REGEX "^#define BB_COMPUTE_TILES ")
string(REGEX REPLACE "^#define BB_COMPUTE_TILES ([0-9]+)$" "\\1" PARTS "${TILE_COUNT_LINE}")
execute_process(COMMAND ${PYTHON} -c
  "import json,sys; from huggingface_hub import hf_hub_download; print(json.load(open(hf_hub_download(sys.argv[1], 'config.json')))['num_hidden_layers'])"
  "${CHECKPOINT}" OUTPUT_VARIABLE LAYERS OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
math(EXPR LAST_LAYER "${LAYERS} - 1")
math(EXPR LAST_RANK "${PARTS} - 1")
set(STAGES)
foreach(PHASE prefill decode)
  list(APPEND STAGES ${PHASE}_embedding)
  foreach(RANK RANGE 0 ${LAST_RANK})
    foreach(LAYER RANGE 0 ${LAST_LAYER})
      list(APPEND STAGES ${PHASE}_attention_${LAYER}_rank_${RANK} ${PHASE}_ffn_${LAYER}_rank_${RANK})
    endforeach()
  endforeach()
endforeach()
list(APPEND STAGES output)
foreach(TARGET attention ffn)
  if(NOT "${TARGET}" IN_LIST TARGETS)
    message(FATAL_ERROR "Llama requires the ${TARGET} core type")
  endif()
endforeach()
set(LLAMA_MLIR)
foreach(STAGE IN LISTS STAGES)
  list(APPEND LLAMA_MLIR ${GENERATED_DIR}/${STAGE}.mlir
                        ${GENERATED_DIR}/${STAGE}-forward.mlir)
endforeach()
file(GLOB_RECURSE DESIGN_FILES CONFIGURE_DEPENDS
  "${DESIGN_DIR}/quant/*.py" "${DESIGN_DIR}/trace/*.py"
  "${DESIGN_DIR}/trace/*.toml" "${DESIGN_DIR}/permute/*.py"
  "${DESIGN_DIR}/loader/*.py")
set(DESIGN_DEPENDS ${DESIGN_DIR}/__init__.py ${DESIGN_FILES}
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/layout/convolution.py
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/layout/matrix.py
  ${REPO_ROOT}/stack/compiler/quant/mxfp8_graph.py
  ${REPO_ROOT}/stack/compiler/quant/mxfp8_embedding.py
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/quantization/mxfp8.py
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/quantization/activation.py
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/quantization/symmetric.py
  ${REPO_ROOT}/stack/compiler/quant/rax.py)
add_subdirectory(trace prepare)
add_subdirectory(${MODEL_SOURCE} model)

set(FORWARD_PASSES
  -buffer-deallocation-simplification -convert-linalg-to-loops
  -eliminate-empty-tensors -llvm-request-c-wrappers
  -convert-math-to-llvm -convert-math-to-libm -lower-affine
  -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
  -expand-strided-metadata -finalize-memref-to-llvm
  -convert-func-to-llvm -reconcile-unrealized-casts -canonicalize -cse
  -lower-tile-runtime)
set(KERNEL_PASSES
  -eliminate-empty-tensors -convert-elementwise-to-linalg
  -convert-tensor-to-linalg -fuse-pointwise
  -outline-rvv-matmul -outline-rvv-kernels -fold-unit-transpose -canonicalize
  -canonicalize -cse
  -one-shot-bufferize=bufferize-function-boundaries -lower-rvv-kernels
  -lower-rvv-pointwise
  -lower-gather-rows -fuse-norm-quant-window
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
set(OBJECTS)
foreach(STAGE IN LISTS STAGES)
  if(STAGE MATCHES "_ffn_" OR STAGE STREQUAL "output")
    set(TARGET ffn)
  else()
    set(TARGET attention)
  endif()
  compile_graph(${STAGE}-forward "${GENERATED_DIR}/${STAGE}-forward.mlir" ${TARGET} forward)
  compile_graph(${STAGE} "${GENERATED_DIR}/${STAGE}.mlir" ${TARGET} kernel)
  list(APPEND OBJECTS ${STAGE}.o ${STAGE}-forward.o)
endforeach()
add_library(llama_kernels STATIC ${OBJECTS})
set_target_properties(llama_kernels PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(llama_kernels llama-model-build)
add_executable(llama-run execution.cpp worker.cpp loader.cpp step.cpp)
add_dependencies(llama-run model-loader)
target_include_directories(llama-run PRIVATE ${MODEL_SOURCE} ${DESIGN_DIR} ${ISA_DIR}/chip ${GENERATED_DIR} ${BUDDY_SOURCE}/frontend/Interfaces)
add_subdirectory("${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/rvv" rvv)
add_subdirectory("${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/mxfp8" mxfp8)
add_subdirectory("${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/mxquant" mxquant)
add_subdirectory("${REPO_ROOT}/examples/balls/mxmm/workloads/kernels" mxmm)
target_link_libraries(llama-run PRIVATE -static llama_model llama_kernels runtime_ant_tasks rvv_kernels_ant mxfp8_pack_ant mxquant_backend_ant mxmm_kernels_ant)
add_custom_command(TARGET llama-run POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${ARTIFACT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:llama-run> ${ARTIFACT_DIR}/llama-run
  COMMAND ${PYTHON} -m ${DESIGN_MODULE}.loader.pack
          --generated-dir ${GENERATED_DIR}
          --model-dir ${ARTIFACT_DIR} --run-config ${DESIGN_DIR}/configs/running-param.toml --backend buckyball --rax-pack ${COMPILER_BUILD}/bin/rax-pack
  WORKING_DIRECTORY ${REPO_ROOT}
  VERBATIM)
set_property(TARGET llama-run APPEND PROPERTY LINK_DEPENDS
  ${DESIGN_DIR}/loader/pack.py ${DESIGN_DIR}/loader/inputs.py ${DESIGN_DIR}/configs/running-param.toml ${REPO_ROOT}/stack/compiler/package.py ${GENERATED_DIR}/model.json)
