set(CMAKE_CXX_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

set(MODEL_QWEN3_DIR ${MODEL_DIR}/Qwen3)
set(INTERFACE_DIR ${BUDDY_MLIR_DIR}/frontend/Interfaces)
set(TOSA_PIPELINE "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))")

add_custom_command(
  OUTPUT forward_prefill_0_6b.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_QWEN3_DIR}/forward_prefill_0_6b.mlir
            -simplify-tosa-reshape |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize="bufferize-function-boundaries"
            -expand-strided-metadata
            -convert-linalg-to-loops
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-cf-to-llvm
            -convert-vector-to-llvm
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard -o forward_prefill_0_6b.o
  DEPENDS ${MODEL_QWEN3_DIR}/forward_prefill_0_6b.mlir
  COMMENT "Building forward_prefill_0_6b.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph_prefill_0_6b.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_QWEN3_DIR}/subgraph0_prefill_0_6b.mlir
            -simplify-tosa-reshape |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-trace-to-llvm="cycle-trace"
            -expand-strided-metadata
            -convert-linalg-to-loops
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-cf-to-llvm
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard -o subgraph_prefill_0_6b.o
  DEPENDS ${MODEL_QWEN3_DIR}/subgraph0_prefill_0_6b.mlir
  COMMENT "Building subgraph_prefill_0_6b.o"
  VERBATIM)

add_custom_command(
  OUTPUT forward_decode_0_6b.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_QWEN3_DIR}/forward_decode_0_6b.mlir
            -simplify-tosa-reshape |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize="bufferize-function-boundaries"
            -expand-strided-metadata
            -convert-linalg-to-loops
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-cf-to-llvm
            -convert-vector-to-llvm
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard -o forward_decode_0_6b.o
  DEPENDS ${MODEL_QWEN3_DIR}/forward_decode_0_6b.mlir
  COMMENT "Building forward_decode_0_6b.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph_decode_0_6b.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_QWEN3_DIR}/subgraph0_decode_0_6b.mlir
            -simplify-tosa-reshape |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -simplify-tosa-matmul-scalar |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-trace-to-llvm="cycle-trace"
            -expand-strided-metadata
            -convert-linalg-to-loops
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-cf-to-llvm
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard -o subgraph_decode_0_6b.o
  DEPENDS ${MODEL_QWEN3_DIR}/subgraph0_decode_0_6b.mlir
  COMMENT "Building subgraph_decode_0_6b.o"
  VERBATIM)

add_library(BuckyballQWEN3_0_6B STATIC
  forward_prefill_0_6b.o
  subgraph_prefill_0_6b.o
  forward_decode_0_6b.o
  subgraph_decode_0_6b.o
)
set_target_properties(BuckyballQWEN3_0_6B PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(BuckyballQWEN3_0_6B qwen3-model-build)

add_executable(buddy-buckyball-qwen3-run ${MODEL_QWEN3_DIR}/buddy-qwen3-0.6b-main.cpp)
add_dependencies(buddy-buckyball-qwen3-run BuckyballQWEN3_0_6B)

target_compile_definitions(buddy-buckyball-qwen3-run PRIVATE
  QWEN3_0_6B_EXAMPLE_PATH="${MODEL_QWEN3_DIR}/"
  QWEN3_0_6B_EXAMPLE_BUILD_PATH="${CMAKE_CURRENT_BINARY_DIR}/"
)
target_include_directories(buddy-buckyball-qwen3-run PRIVATE ${INTERFACE_DIR})
set(BUDDY_QWEN3_LIBS BuckyballQWEN3_0_6B ModelTestCRunnerUtils)
target_link_libraries(buddy-buckyball-qwen3-run -static ${BUDDY_QWEN3_LIBS})

set(QWEN3_OUTPUT_DIR ${OUTPUT_BIN_DIR}/src/ModelTest/e2e/models/archs/buckyball/Qwen3)

add_custom_command(TARGET buddy-buckyball-qwen3-run POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${QWEN3_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:buddy-buckyball-qwen3-run>
          ${QWEN3_OUTPUT_DIR}/buddy-buckyball-qwen3-run
  COMMENT "Copying Buckyball Qwen3 executable to output directory"
  VERBATIM
)

add_custom_target(sync-buckyball-qwen3-deps
  COMMAND ${CMAKE_COMMAND} -E make_directory ${QWEN3_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${QWEN3_OUTPUT_DIR}/trace/cycle
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_QWEN3_DIR}/arg0_0_6b.data
          ${CMAKE_CURRENT_BINARY_DIR}/arg0_0_6b.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_QWEN3_DIR}/arg1_0_6b.data
          ${CMAKE_CURRENT_BINARY_DIR}/arg1_0_6b.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_QWEN3_DIR}/arg0_0_6b.data
          ${QWEN3_OUTPUT_DIR}/arg0_0_6b.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_QWEN3_DIR}/arg1_0_6b.data
          ${QWEN3_OUTPUT_DIR}/arg1_0_6b.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_QWEN3_DIR}/vocab.txt
          ${QWEN3_OUTPUT_DIR}/vocab.txt
  DEPENDS
          ${MODEL_QWEN3_DIR}/arg0_0_6b.data
          ${MODEL_QWEN3_DIR}/arg1_0_6b.data
          ${MODEL_QWEN3_DIR}/vocab.txt
  COMMENT "Syncing Qwen3 runtime deps to output directory"
  VERBATIM
)

add_dependencies(buddy-buckyball-qwen3-run sync-buckyball-qwen3-deps)
