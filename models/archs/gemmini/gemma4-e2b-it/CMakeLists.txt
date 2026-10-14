set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

add_custom_command(
  OUTPUT ${OUTPUT_DIR}/forward_prefill.mlir
         ${OUTPUT_DIR}/subgraph0_prefill.mlir
         ${OUTPUT_DIR}/forward_decode.mlir
         ${OUTPUT_DIR}/subgraph0_decode.mlir
         ${OUTPUT_DIR}/arg0_prefill.data
         ${OUTPUT_DIR}/arg0_decode.data
         ${OUTPUT_DIR}/vocab.txt
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
  COMMAND ${PYTHON} buddy-pytorch-importer.py
            --repo-root ${REPO_ROOT}
            --output-dir ${OUTPUT_DIR}
            --compiler-build ${COMPILER_BUILD}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing gemma4-e2b-it for gemmini"
  VERBATIM
)

add_custom_command(
  OUTPUT forward_prefill.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_prefill_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_prefill.mlir
            -simplify-tosa-reshape |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize="bufferize-function-boundaries"
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/forward_prefill_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_prefill.mlir
            -simplify-tosa-reshape |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize="bufferize-function-boundaries"
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${COMPILER_BUILD}/bin/buddy-translate
            --buddy-to-llvmir |
          ${COMPILER_BUILD}/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xgemmini,+D -float-abi=hard -o forward_prefill.o
  DEPENDS ${OUTPUT_DIR}/forward_prefill.mlir
  COMMENT "Building forward_prefill.o"
  VERBATIM
)

add_custom_command(
  OUTPUT subgraph_prefill.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph_prefill_gemmini.mlir
             ${OUTPUT_DIR}/subgraph_prefill_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_prefill.mlir
            -simplify-tosa-reshape |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-gemmini
            -o ${OUTPUT_DIR}/subgraph_prefill_gemmini.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_prefill.mlir
            -simplify-tosa-reshape |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -lower-gemmini
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/subgraph_prefill_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_prefill.mlir
            -simplify-tosa-reshape |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -lower-gemmini
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${COMPILER_BUILD}/bin/buddy-translate
            --buddy-to-llvmir |
          ${COMPILER_BUILD}/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xgemmini,+D -float-abi=hard -o subgraph_prefill.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_prefill.mlir
  COMMENT "Building subgraph_prefill.o"
  VERBATIM
)

add_custom_command(
  OUTPUT forward_decode.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_decode_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_decode.mlir
            -simplify-tosa-reshape |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize="bufferize-function-boundaries"
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/forward_decode_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_decode.mlir
            -simplify-tosa-reshape |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize="bufferize-function-boundaries"
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${COMPILER_BUILD}/bin/buddy-translate
            --buddy-to-llvmir |
          ${COMPILER_BUILD}/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xgemmini,+D -float-abi=hard -o forward_decode.o
  DEPENDS ${OUTPUT_DIR}/forward_decode.mlir
  COMMENT "Building forward_decode.o"
  VERBATIM
)

add_custom_command(
  OUTPUT subgraph_decode.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph_decode_gemmini.mlir
             ${OUTPUT_DIR}/subgraph_decode_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_decode.mlir
            -simplify-tosa-reshape |
          ${COMPILER_BUILD}/bin/buddy-opt
            -simplify-tosa-matmul-scalar |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-gemmini
            -o ${OUTPUT_DIR}/subgraph_decode_gemmini.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_decode.mlir
            -simplify-tosa-reshape |
          ${COMPILER_BUILD}/bin/buddy-opt
            -simplify-tosa-matmul-scalar |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -lower-gemmini
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/subgraph_decode_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_decode.mlir
            -simplify-tosa-reshape |
          ${COMPILER_BUILD}/bin/buddy-opt
            -simplify-tosa-matmul-scalar |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -lower-gemmini
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${COMPILER_BUILD}/bin/buddy-translate
            --buddy-to-llvmir |
          ${COMPILER_BUILD}/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xgemmini,+D -float-abi=hard -o subgraph_decode.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_decode.mlir
  COMMENT "Building subgraph_decode.o"
  VERBATIM
)

add_library(gemma4-e2b-it-gemmini-model STATIC EXCLUDE_FROM_ALL forward_prefill.o subgraph_prefill.o forward_decode.o
  subgraph_decode.o
)
set_target_properties(gemma4-e2b-it-gemmini-model PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
set_target_properties(gemma4-e2b-it-gemmini-model PROPERTIES LINKER_LANGUAGE CXX)

add_executable(buddy-gemmini-gemma4-run EXCLUDE_FROM_ALL ${MODEL_DIR}/gemma4-e2b-it/main.cpp)
set_target_properties(buddy-gemmini-gemma4-run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
add_dependencies(buddy-gemmini-gemma4-run gemma4-e2b-it-gemmini-model)

target_include_directories(buddy-gemmini-gemma4-run PRIVATE ${BUDDY_SOURCE}/frontend/Interfaces)
target_link_libraries(buddy-gemmini-gemma4-run -static gemma4-e2b-it-gemmini-model
                                                       gemmini-runtime
)

add_custom_target(gemma4-e2b-it-model-build DEPENDS gemma4-e2b-it-gemmini-model)
