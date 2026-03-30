set(CMAKE_CXX_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

set(MODEL_QWEN3_DIR ${MODEL_DIR}/Qwen3)
set(INTERFACE_DIR ${BUDDY_MLIR_DIR}/frontend/Interfaces)
set(TOSA_PIPELINE "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))")

add_custom_command(
  OUTPUT forward_prefill_0_6b.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_QWEN3_DIR}/forward_prefill_0_6b.mlir
            -simplify-tosa-reshape |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_BINARY_DIR}/buddy-opt
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
          ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard -o forward_prefill_0_6b.o
  DEPENDS ${MODEL_QWEN3_DIR}/forward_prefill_0_6b.mlir
  COMMENT "Building forward_prefill_0_6b.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph_prefill_0_6b.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_QWEN3_DIR}/subgraph0_prefill_0_6b.mlir
            -simplify-tosa-reshape |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
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
          ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard -o subgraph_prefill_0_6b.o
  DEPENDS ${MODEL_QWEN3_DIR}/subgraph0_prefill_0_6b.mlir
  COMMENT "Building subgraph_prefill_0_6b.o"
  VERBATIM)

add_custom_command(
  OUTPUT forward_decode_0_6b.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_QWEN3_DIR}/forward_decode_0_6b.mlir
            -simplify-tosa-reshape |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_BINARY_DIR}/buddy-opt
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
          ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard -o forward_decode_0_6b.o
  DEPENDS ${MODEL_QWEN3_DIR}/forward_decode_0_6b.mlir
  COMMENT "Building forward_decode_0_6b.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph_decode_0_6b.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_QWEN3_DIR}/subgraph0_decode_0_6b.mlir
            -simplify-tosa-reshape
            -simplify-tosa-matmul-scalar |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
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
          ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard -o subgraph_decode_0_6b.o
  DEPENDS ${MODEL_QWEN3_DIR}/subgraph0_decode_0_6b.mlir
  COMMENT "Building subgraph_decode_0_6b.o"
  VERBATIM)

add_library(GemminiQWEN3_0_6B STATIC
  forward_prefill_0_6b.o
  subgraph_prefill_0_6b.o
  forward_decode_0_6b.o
  subgraph_decode_0_6b.o
)
set_target_properties(GemminiQWEN3_0_6B PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(GemminiQWEN3_0_6B qwen3-model-build)

add_executable(buddy-gemmini-qwen3-run ${MODEL_QWEN3_DIR}/buddy-qwen3-0.6b-main.cpp)
add_dependencies(buddy-gemmini-qwen3-run GemminiQWEN3_0_6B)

target_compile_definitions(buddy-gemmini-qwen3-run PRIVATE
  QWEN3_0_6B_EXAMPLE_PATH="./"
  QWEN3_0_6B_EXAMPLE_BUILD_PATH="./"
)
target_include_directories(buddy-gemmini-qwen3-run PRIVATE ${INTERFACE_DIR})
set(BUDDY_QWEN3_LIBS GemminiQWEN3_0_6B ModelTestCRunnerUtils)
target_link_libraries(buddy-gemmini-qwen3-run ${BUDDY_QWEN3_LIBS})
