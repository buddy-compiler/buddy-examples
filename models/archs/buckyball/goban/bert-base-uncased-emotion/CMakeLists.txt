set(TRACE_SUPPORTED TRUE PARENT_SCOPE)
set(DESIGN_MODULE examples.models.goban.bert-base-uncased-emotion)
set(LLVM_OPT_LEVEL 2 CACHE STRING "Backend optimization level")
set(LLVM_PASSES "function(sroa,early-cse,instcombine,simplifycfg,adce)")
set(MATTR "+xbuckyball,+m,+D")
set(MODEL_SOURCE "${REPO_ROOT}/stack/models/models/bert-base-uncased-emotion")
set(PARTITIONS 4 CACHE STRING "BERT subgraphs")
set(TARGET goban)
option(CTRACE "Enable cycle trace" OFF)
option(DTRACE "Enable tensor data trace" OFF)
set(TRACE_ARGS)
set(TRACE_MODES)
set(TRACE_PASS)
if(CTRACE OR DTRACE)
  list(APPEND TRACE_ARGS --trace)
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
if(NOT "${TARGET}" IN_LIST TARGETS)
  message(FATAL_ERROR "Unknown BERT compiler target: ${TARGET}")
endif()
math(EXPR LAST_PARTITION "${PARTITIONS} - 1")
set(BERT_SUBGRAPH_MLIR)
foreach(BERT_PARTITION RANGE 0 ${LAST_PARTITION})
  list(APPEND BERT_SUBGRAPH_MLIR
       ${GENERATED_DIR}/subgraph${BERT_PARTITION}.mlir)
endforeach()

file(GLOB_RECURSE DESIGN_FILES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/quant/*" "${CMAKE_CURRENT_SOURCE_DIR}/trace/*"
  "${CMAKE_CURRENT_SOURCE_DIR}/permute/*")
set(DESIGN_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py ${DESIGN_FILES}
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/layout/convolution.py
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/layout/matrix.py
  ${REPO_ROOT}/stack/compiler/quant/importer.py
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
  -convert-func-to-llvm -reconcile-unrealized-casts -canonicalize -cse -lower-tile-runtime)
set(KERNEL_PASSES
  -eliminate-empty-tensors -convert-elementwise-to-linalg
  -convert-tensor-to-linalg -one-shot-bufferize=bufferize-function-boundaries
  -convert-linalg-to-tile=quantized-only
  -convert-tile-to-buckyball
  -lower-buckyball-to-bank-ssa -assign-physical-banks
  -llvm-request-c-wrappers -lower-bank-ssa-to-intrinsics
  ${TRACE_PASS}
  -expand-strided-metadata -convert-linalg-to-loops
  -lower-buckyball -lower-affine -canonicalize -cse -convert-scf-to-cf
  -convert-cf-to-llvm -buffer-deallocation-simplification
  -bufferization-lower-deallocations -convert-math-to-llvm
  -convert-math-to-libm -convert-arith-to-llvm -convert-func-to-llvm
  -finalize-memref-to-llvm -reconcile-unrealized-casts -canonicalize -cse -lower-tile-runtime)
compile_graph(forward "${GENERATED_DIR}/forward.mlir" ${TARGET} forward)
set(OBJECTS forward.o)
foreach(PARTITION RANGE 0 ${LAST_PARTITION})
  compile_graph(subgraph${PARTITION} "${GENERATED_DIR}/subgraph${PARTITION}.mlir" ${TARGET} kernel)
  list(APPEND OBJECTS subgraph${PARTITION}.o)
endforeach()
add_library(bert_kernels STATIC ${OBJECTS})
set_target_properties(bert_kernels PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(bert_kernels bert-model-build)
add_executable(bert-run ${MODEL_SOURCE}/main.cpp)
add_dependencies(bert-run model-loader)
target_include_directories(bert-run PRIVATE ${GENERATED_DIR} ${BUDDY_SOURCE}/frontend/Interfaces)
target_link_libraries(bert-run PRIVATE -static bert_kernels runtime)
add_custom_command(TARGET bert-run POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${ARTIFACT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:bert-run> ${ARTIFACT_DIR}/bert-run
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${GENERATED_DIR}/bert.payload ${ARTIFACT_DIR}/bert.payload
  COMMAND ${PYTHON} -m stack.models.models.bert-base-uncased-emotion.pack
          --model-dir ${ARTIFACT_DIR} --entrypoint examples.models.goban.bert-base-uncased-emotion.serve:run --backend buckyball --rax-pack ${COMPILER_BUILD}/bin/rax-pack
  WORKING_DIRECTORY ${REPO_ROOT}
  VERBATIM)
set_property(TARGET bert-run APPEND PROPERTY LINK_DEPENDS
  ${MODEL_SOURCE}/pack.py ${REPO_ROOT}/stack/compiler/package.py ${GENERATED_DIR}/bert.rax)
