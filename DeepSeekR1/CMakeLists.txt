add_custom_command(
  OUTPUT 
    ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir 
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
    ${CMAKE_CURRENT_BINARY_DIR}/arg0-deepseek-r1.data
  COMMAND
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/import-deepseek-r1.py
      --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating forward.mlir, subgraph0.mlir and arg0.data..."
)

set(BIN_DIR ${CMAKE_CURRENT_BINARY_DIR}/../bin)
add_custom_command(
  OUTPUT 
    ${BIN_DIR}/arg0-deepseek-r1.data
    ${BIN_DIR}/vocab-deepseek-r1.txt
  COMMAND ${CMAKE_COMMAND} -E rename
    ${CMAKE_CURRENT_BINARY_DIR}/arg0-deepseek-r1.data
    ${BIN_DIR}/arg0-deepseek-r1.data
  COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/vocab-deepseek-r1.txt
    ${BIN_DIR}/
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/vocab-deepseek-r1.txt
  COMMENT "Preparing arg0-deepseek-r1.data and vocab-deepseek-r1.txt..."
)
add_custom_target(deepseek_data
  DEPENDS 
    ${BIN_DIR}/arg0-deepseek-r1.data
    ${BIN_DIR}/vocab-deepseek-r1.txt
)

add_executable(deepseek-r1
  Main.cpp
)
add_dependencies(deepseek-r1
  deepseek_data
)
target_link_libraries(deepseek-r1
  GoogleBenchmark
  ${LLVM_LIBRARY_DIR}
)

separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

################################################################################
#
# Build buddy optimization target.
#
################################################################################
add_custom_command(
  OUTPUT forward_buddy.o
  COMMAND 
    cat ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir |
    sed -e {s/@forward/@forward_buddy/} |
    sed -e {s/@subgraph0/@subgraph0_buddy/} |
    ${LLVM_MLIR_BINARY_DIR}/mlir-opt
      -expand-strided-metadata
      -finalize-memref-to-llvm
      -llvm-request-c-wrappers
      -convert-func-to-llvm
      -reconcile-unrealized-casts |
    ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o forward_buddy.ll
  COMMAND 
    ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} forward_buddy.ll
      -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/forward_buddy.o
  DEPENDS 
    ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
  COMMENT "Building forward_buddy.o"
  VERBATIM
)

add_custom_command(
  OUTPUT subgraph0_buddy.o
  COMMAND 
    cat ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir |
    sed -e {s/@subgraph0/@subgraph0_buddy/} |
    ${LLVM_MLIR_BINARY_DIR}/mlir-opt
      -pass-pipeline 
      "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
    ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
      -convert-elementwise-to-linalg
      -arith-expand
      -eliminate-empty-tensors
      -empty-tensor-to-alloc-tensor
      -one-shot-bufferize
      -matmul-parallel-vectorization-optimize
      -batchmatmul-optimize
      -convert-linalg-to-affine-loops
      -affine-loop-fusion
      -affine-parallelize
      -lower-affine
      -convert-scf-to-openmp
      -func-bufferize-dynamic-offset
      -tensor-bufferize
      -arith-bufferize
      -buffer-deallocation
      -finalizing-bufferize
      -convert-vector-to-scf
      -expand-strided-metadata
      -cse
      -convert-vector-to-llvm
      -memref-expand
      -arith-expand
      -convert-arith-to-llvm
      -finalize-memref-to-llvm
      -convert-scf-to-cf
      -llvm-request-c-wrappers
      -convert-openmp-to-llvm
      -convert-arith-to-llvm
      -convert-math-to-llvm
      -convert-math-to-libm 
      -convert-func-to-llvm
      -reconcile-unrealized-casts |
    ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o subgraph0_buddy.ll
    COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} subgraph0_buddy.ll
      -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_buddy.o
    DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
  COMMENT "Building subgraph0_buddy.o"
  VERBATIM
)
add_library(deepseekr1_buddy STATIC subgraph0_buddy.o forward_buddy.o)
set_target_properties(deepseekr1_buddy PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(deepseek-r1
  deepseekr1_buddy
  ${BUDDY_LIB_DIR}/libStaticMLIRCRunnerUtils.a
  ${OMP_SHARED_DIR}/libomp.so
)
