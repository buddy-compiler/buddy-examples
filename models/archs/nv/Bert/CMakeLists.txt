# Host x86 + NVIDIA GPU path (copied from nv/LeNet, adapted).

set(MODEL_BERT_DIR ${MODEL_DIR}/Bert)
set(INTERFACE_DIR ${NV_INTERFACE_DIR})
set(INCLUDE_DIR ${MODEL_BERT_DIR})


# forward.mlir is already memref glue (like BuddyBert); keep it on host CPU.
add_custom_command(
  OUTPUT forward_gpu.o
  COMMAND ${NV_LLVM_TOOLS_BINARY_DIR}/mlir-opt ${MODEL_BERT_DIR}/forward.mlir
          -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg)" |
          ${NV_LLVM_TOOLS_BINARY_DIR}/mlir-opt
          -pass-pipeline "builtin.module(func.func(buffer-deallocation-simplification, convert-linalg-to-loops), eliminate-empty-tensors, func.func(llvm-request-c-wrappers), convert-math-to-llvm, convert-math-to-libm, convert-scf-to-cf, convert-cf-to-llvm, convert-arith-to-llvm, expand-strided-metadata, finalize-memref-to-llvm, convert-func-to-llvm, reconcile-unrealized-casts)" |
          ${NV_LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${NV_LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${NV_LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O0
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_gpu.o
  DEPENDS ${MODEL_BERT_DIR}/forward.mlir
  COMMENT "Building NV forward_gpu.o"
  VERBATIM)


# Bert GPU path is broken: process-args=0 passes host strided memrefs into
# gpu.launch_func (CUDA_ERROR_ILLEGAL_ADDRESS); process-args=1 breaks subview
# layouts. Lower subgraph on host CPU (sequential, no OpenMP) until that is fixed.
add_custom_command(
  OUTPUT subgraph0_gpu.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_BERT_DIR}/subgraph0.mlir
          -convert-trace-to-llvm=cycle-trace |
          ${NV_LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
          -convert-elementwise-to-linalg
          -arith-expand
          -eliminate-empty-tensors
          -empty-tensor-to-alloc-tensor
          -one-shot-bufferize="bufferize-function-boundaries"
          -ownership-based-buffer-deallocation
          -buffer-deallocation-simplification
          -bufferization-lower-deallocations
          -matmul-parallel-vectorization-optimize
          -convert-linalg-to-affine-loops
          -affine-loop-fusion
          -lower-affine
          -convert-vector-to-scf
          -expand-strided-metadata
          -lower-affine
          -cse
          -convert-vector-to-llvm
          -memref-expand
          -arith-expand
          -convert-arith-to-llvm
          -finalize-memref-to-llvm
          -convert-scf-to-cf
          -convert-cf-to-llvm
          -llvm-request-c-wrappers
          -convert-arith-to-llvm
          -convert-math-to-llvm
          -convert-math-to-libm
          -convert-func-to-llvm
          -reconcile-unrealized-casts |
          ${NV_LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${NV_LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${NV_LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O0
          -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_gpu.o
  DEPENDS ${MODEL_BERT_DIR}/subgraph0.mlir
          ${BUDDY_BINARY_DIR}/buddy-opt
  COMMENT "Building NV subgraph0_gpu.o (host CPU, no OpenMP)"
  VERBATIM)

add_library(NvBERT STATIC subgraph0_gpu.o forward_gpu.o)
set_target_properties(NvBERT PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(NvBERT bert-model-build)

add_executable(buddy-nv-bert-run ${MODEL_BERT_DIR}/bert-main.cpp)
add_dependencies(buddy-nv-bert-run NvBERT)
target_include_directories(buddy-nv-bert-run PRIVATE ${INTERFACE_DIR} ${INCLUDE_DIR})
target_link_directories(buddy-nv-bert-run PRIVATE ${NV_LLVM_LIBRARY_DIR} ${BUDDY_MLIR_DIR}/build/lib)
target_link_libraries(buddy-nv-bert-run
  NvBERT
  NvHostCRunnerUtils
  PNG::PNG
  ${NV_BUDDY_LIB_DIP}
  ${NV_MLIR_C_RUNNER_UTILS_LIB}
  ${NV_MLIR_RUNNER_UTILS_LIB}
  ${NV_MLIR_ASYNC_RUNTIME_LIB}
  ${NV_MLIR_CUDA_RUNTIME_LIB}
  dl
  pthread
)

add_custom_target(sync-nv-bert-deps
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/trace/cycle
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_BERT_DIR}/arg0.data
          ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_BERT_DIR}/arg1.data
          ${CMAKE_CURRENT_BINARY_DIR}/arg1.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_BERT_DIR}/vocab.txt
          ${CMAKE_CURRENT_BINARY_DIR}/vocab.txt
  DEPENDS
          ${MODEL_BERT_DIR}/arg0.data
          ${MODEL_BERT_DIR}/arg1.data
          ${MODEL_BERT_DIR}/vocab.txt
  COMMENT "Syncing NV bert runtime deps"
  VERBATIM)
add_dependencies(buddy-nv-bert-run sync-nv-bert-deps)

if(DEFINED OUTPUT_BIN_DIR)
  set(OUT_DIR ${OUTPUT_BIN_DIR}/src/ModelTest/e2e/models/archs/nv/Bert)
  add_custom_command(TARGET buddy-nv-bert-run POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}/trace/cycle
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:buddy-nv-bert-run>
            ${OUT_DIR}/buddy-nv-bert-run
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${MODEL_BERT_DIR}/arg0.data
            ${OUT_DIR}/arg0.data
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${MODEL_BERT_DIR}/arg1.data
            ${OUT_DIR}/arg1.data
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${MODEL_BERT_DIR}/vocab.txt
            ${OUT_DIR}/vocab.txt
    COMMENT "Copying NV bert executable and deps to output directory"
    VERBATIM)
endif()

add_custom_target(buddy-nv-bert-all-run DEPENDS buddy-nv-bert-run)
