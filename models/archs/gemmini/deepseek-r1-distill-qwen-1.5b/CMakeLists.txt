set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

add_custom_command(
  OUTPUT ${OUTPUT_DIR}/forward.mlir
         ${OUTPUT_DIR}/subgraph0.mlir
         ${OUTPUT_DIR}/arg0.data
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
  COMMAND ${PYTHON} buddy-pytorch-importer.py
            --repo-root ${REPO_ROOT}
            --output-dir ${OUTPUT_DIR}
            --compiler-build ${COMPILER_BUILD}
            --checkpoint "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing deepseek-r1-distill-qwen-1.5b for gemmini"
  VERBATIM
)

add_custom_command(
  OUTPUT forward.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${BUDDY_SOURCE}/build/bin/buddy-opt ${OUTPUT_DIR}/forward.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_SOURCE}/build/bin/buddy-opt
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -lower-affine
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-vector-to-scf
            -expand-strided-metadata
            -convert-vector-to-llvm
            -memref-expand
            -arith-expand
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${BUDDY_SOURCE}/build/bin/buddy-opt ${OUTPUT_DIR}/forward.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_SOURCE}/build/bin/buddy-opt
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -lower-affine
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-vector-to-scf
            -expand-strided-metadata
            -convert-vector-to-llvm
            -memref-expand
            -arith-expand
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${BUDDY_SOURCE}/build/bin/buddy-translate
            --buddy-to-llvmir |
          ${BUDDY_SOURCE}/build/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xgemmini,+D -float-abi=hard -o forward.o
  DEPENDS ${OUTPUT_DIR}/forward.mlir
  COMMENT "Building forward.o "
  VERBATIM
)

add_custom_command(
  OUTPUT subgraph0.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_gemmini.mlir
             ${OUTPUT_DIR}/subgraph0_test.mlir
             ${OUTPUT_DIR}/subgraph0.ll
             ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${BUDDY_SOURCE}/build/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_SOURCE}/build/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-gemmini
            -o ${OUTPUT_DIR}/subgraph0_gemmini.mlir
  COMMAND ${BUDDY_SOURCE}/build/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_SOURCE}/build/bin/buddy-opt
            -eliminate-empty-tensors
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
            -convert-cf-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_SOURCE}/build/bin/buddy-opt
            -convert-tensor-to-linalg
            -o ${OUTPUT_DIR}/subgraph0_linalg.mlir
  COMMAND ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_SOURCE}/build/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-gemmini
            -lower-gemmini
            -o ${OUTPUT_DIR}/subgraph0_gemmini.mlir
  COMMAND ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_SOURCE}/build/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-gemmini
            -lower-gemmini
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -expand-strided-metadata
            -llvm-request-c-wrappers
            -o ${OUTPUT_DIR}/subgraph0_test.mlir
  COMMAND ${BUDDY_SOURCE}/build/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_SOURCE}/build/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-gemmini
            -convert-linalg-to-loops
            -expand-strided-metadata
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -lower-gemmini
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-cf-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${BUDDY_SOURCE}/build/bin/buddy-translate
            --buddy-to-llvmir -o ${OUTPUT_DIR}/subgraph0.ll
  COMMAND ${BUDDY_SOURCE}/build/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_SOURCE}/build/bin/buddy-opt
            -eliminate-empty-tensors
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
            -convert-cf-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${BUDDY_SOURCE}/build/bin/buddy-translate
            --buddy-to-llvmir |
          ${BUDDY_SOURCE}/build/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O3 -mattr=+xgemmini,+D -float-abi=hard -o subgraph0.o
  DEPENDS ${OUTPUT_DIR}/subgraph0.mlir
  COMMENT "Building subgraph0.o"
  VERBATIM
)

add_library(deepseek-r1-distill-qwen-1.5b-gemmini-model STATIC EXCLUDE_FROM_ALL forward.o subgraph0.o)
set_target_properties(deepseek-r1-distill-qwen-1.5b-gemmini-model PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
set_target_properties(deepseek-r1-distill-qwen-1.5b-gemmini-model PROPERTIES LINKER_LANGUAGE CXX)
add_executable(buddy-gemmini-deepseekr1-run EXCLUDE_FROM_ALL ${MODEL_DIR}/deepseek-r1-distill-qwen-1.5b/main.cpp)
set_target_properties(buddy-gemmini-deepseekr1-run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
add_dependencies(buddy-gemmini-deepseekr1-run deepseek-r1-distill-qwen-1.5b-gemmini-model)

target_include_directories(buddy-gemmini-deepseekr1-run PRIVATE ${BUDDY_SOURCE}/frontend/Interfaces
                                                                ${GEMMINI_DIR}/deepseek-r1-distill-qwen-1.5b/include
)

if(BUDDY_MLIR_USE_MIMALLOC)
  list(APPEND GEMMINI_DEEPSEEKR1_LIBS mimalloc)
endif()

target_link_libraries(buddy-gemmini-deepseekr1-run -static deepseek-r1-distill-qwen-1.5b-gemmini-model
                                                           gemmini-runtime
)

add_custom_target(deepseek-r1-distill-qwen-1.5b-model-build DEPENDS deepseek-r1-distill-qwen-1.5b-gemmini-model)
