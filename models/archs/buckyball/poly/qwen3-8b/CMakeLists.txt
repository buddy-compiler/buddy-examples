set(DESIGN_MODULE examples.models.poly.qwen3-8b)
set(CHECKPOINT "Qwen/Qwen3-8B")
set(DESIGN_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(SERVING_ENTRYPOINT examples.models.poly.qwen3-8b.serve:run)
set(LLVM_OPT_LEVEL 2 CACHE STRING "Backend optimization level")
set(LLVM_PASSES "function(sroa,early-cse,instcombine,simplifycfg,adce)")
set(MATTR "+xbuckyball,+m,+D")
set(MODEL_SOURCE "${REPO_ROOT}/stack/models/models/qwen3-8b")
set(PREFILL_LENGTH 8 CACHE STRING "Compiled prefill capacity")
set(CACHE_LENGTH 16 CACHE STRING "Compiled KV capacity")
set(PARTS 4 CACHE STRING "Tensor parallel tile count")
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
    message(FATAL_ERROR "Qwen requires the ${TARGET} core type")
  endif()
endforeach()
set(QWEN_MLIR)
foreach(STAGE IN LISTS STAGES)
  list(APPEND QWEN_MLIR ${GENERATED_DIR}/${STAGE}.mlir
                        ${GENERATED_DIR}/${STAGE}-forward.mlir)
endforeach()
file(GLOB_RECURSE DESIGN_FILES CONFIGURE_DEPENDS
  "${DESIGN_DIR}/quant/*" "${DESIGN_DIR}/trace/*"
  "${DESIGN_DIR}/permute/*")
set(DESIGN_DEPENDS ${DESIGN_DIR}/__init__.py ${DESIGN_FILES}
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/layout/convolution.py
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/layout/matrix.py
  ${REPO_ROOT}/stack/compiler/quant/mxfp8_graph.py
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/quantization/mxfp8.py
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/quantization/activation.py
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/quantization/symmetric.py
  ${REPO_ROOT}/stack/compiler/quant/rax.py)
add_subdirectory(${MODEL_SOURCE} prepare)

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
  -convert-tensor-to-linalg -one-shot-bufferize=bufferize-function-boundaries
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
  -lower-tile-runtime)
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
add_library(qwen_kernels STATIC ${OBJECTS})
set_target_properties(qwen_kernels PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(qwen_kernels qwen-model-build)
add_executable(qwen-run ${MODEL_SOURCE}/main.cpp)
add_dependencies(qwen-run model-loader)
target_include_directories(qwen-run PRIVATE ${GENERATED_DIR} ${BUDDY_SOURCE}/frontend/Interfaces)
target_link_libraries(qwen-run PRIVATE -static qwen_kernels runtime)
add_custom_command(TARGET qwen-run POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${ARTIFACT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:qwen-run> ${ARTIFACT_DIR}/qwen-run
  COMMAND ${PYTHON} -m stack.models.models.qwen3-8b.pack
          --generated-dir ${GENERATED_DIR}
          --model-dir ${ARTIFACT_DIR} --entrypoint ${SERVING_ENTRYPOINT} --backend buckyball --rax-pack ${COMPILER_BUILD}/bin/rax-pack
  WORKING_DIRECTORY ${REPO_ROOT}
  VERBATIM)
set_property(TARGET qwen-run APPEND PROPERTY LINK_DEPENDS
  ${MODEL_SOURCE}/pack.py ${REPO_ROOT}/stack/compiler/package.py ${GENERATED_DIR}/model.json)
