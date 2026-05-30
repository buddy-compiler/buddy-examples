set(CMAKE_CXX_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

set(MODEL_GEMMA4_DIR ${MODEL_DIR}/Gemma4)
set(INTERFACE_DIR ${BUDDY_MLIR_DIR}/frontend/Interfaces)
set(TOSA_PIPELINE "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))")

add_custom_command(
  OUTPUT forward_prefill.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_GEMMA4_DIR}/forward_prefill.mlir
            -simplify-tosa-reshape |
          ${BUDDY_BINARY_DIR}/buddy-opt
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
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard -o forward_prefill.o
  DEPENDS ${MODEL_GEMMA4_DIR}/forward_prefill.mlir
  COMMENT "Building forward_prefill.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph_prefill.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_GEMMA4_DIR}/subgraph0_prefill.mlir
            -simplify-tosa-reshape |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-tile
            -convert-tile-to-buckyball
            -lower-buckyball-to-bank-ssa
            -assign-physical-banks
            -llvm-request-c-wrappers
            -lower-bank-ssa-to-intrinsics
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-buckyball
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
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard -o subgraph_prefill.o
  DEPENDS ${MODEL_GEMMA4_DIR}/subgraph0_prefill.mlir
  COMMENT "Building subgraph_prefill.o"
  VERBATIM)

add_custom_command(
  OUTPUT forward_decode.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_GEMMA4_DIR}/forward_decode.mlir
            -simplify-tosa-reshape |
          ${BUDDY_BINARY_DIR}/buddy-opt
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
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard -o forward_decode.o
  DEPENDS ${MODEL_GEMMA4_DIR}/forward_decode.mlir
  COMMENT "Building forward_decode.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph_decode.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_GEMMA4_DIR}/subgraph0_decode.mlir
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
            -convert-linalg-to-tile
            -convert-tile-to-buckyball
            -lower-buckyball-to-bank-ssa
            -assign-physical-banks
            -llvm-request-c-wrappers
            -lower-bank-ssa-to-intrinsics
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-buckyball
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
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard -o subgraph_decode.o
  DEPENDS ${MODEL_GEMMA4_DIR}/subgraph0_decode.mlir
  COMMENT "Building subgraph_decode.o"
  VERBATIM)

add_library(BuckyballGEMMA4 STATIC
  forward_prefill.o
  subgraph_prefill.o
  forward_decode.o
  subgraph_decode.o
)
set_target_properties(BuckyballGEMMA4 PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(BuckyballGEMMA4 gemma4-model-build)

add_executable(buddy-buckyball-gemma4-run ${MODEL_GEMMA4_DIR}/buddy-gemma4-main.cpp)
add_dependencies(buddy-buckyball-gemma4-run BuckyballGEMMA4)

target_compile_definitions(buddy-buckyball-gemma4-run PRIVATE
  GEMMA4_EXAMPLE_PATH="${MODEL_GEMMA4_DIR}/"
  GEMMA4_EXAMPLE_BUILD_PATH="${CMAKE_CURRENT_BINARY_DIR}/"
)
target_include_directories(buddy-buckyball-gemma4-run PRIVATE ${INTERFACE_DIR})
set(BUDDY_GEMMA4_LIBS BuckyballGEMMA4 ModelTestCRunnerUtils)
target_link_libraries(buddy-buckyball-gemma4-run -static ${BUDDY_GEMMA4_LIBS})

set(GEMMA4_OUTPUT_DIR ${OUTPUT_BIN_DIR}/src/ModelTest/e2e/models/archs/buckyball/Gemma4)

add_custom_command(TARGET buddy-buckyball-gemma4-run POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GEMMA4_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:buddy-buckyball-gemma4-run>
          ${GEMMA4_OUTPUT_DIR}/buddy-buckyball-gemma4-run
  COMMENT "Copying Buckyball Gemma4 executable to output directory"
  VERBATIM
)

add_custom_target(sync-buckyball-gemma4-deps
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GEMMA4_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_GEMMA4_DIR}/arg0.data
          ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_GEMMA4_DIR}/arg0.data
          ${GEMMA4_OUTPUT_DIR}/arg0.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_GEMMA4_DIR}/vocab.txt
          ${CMAKE_CURRENT_BINARY_DIR}/vocab.txt
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_GEMMA4_DIR}/vocab.txt
          ${GEMMA4_OUTPUT_DIR}/vocab.txt
  DEPENDS
          ${MODEL_GEMMA4_DIR}/arg0.data
          ${MODEL_GEMMA4_DIR}/vocab.txt
  COMMENT "Syncing Gemma4 runtime deps to output directory"
  VERBATIM
)

add_dependencies(buddy-buckyball-gemma4-run sync-buckyball-gemma4-deps)
