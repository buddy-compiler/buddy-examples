set(CMAKE_CXX_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

set(MODEL_LLAMA2_DIR ${MODEL_DIR}/llama2)
set(BUCKYBALL_LLAMA2_DIR ${BUCKYBALL_DIR}/llama2)
set(INTERFACE_DIR ${BUDDY_MLIR_DIR}/frontend/Interfaces)
set(INCLUDE_DIR ${MODEL_LLAMA2_DIR}/include)

add_custom_command(
  OUTPUT forward.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_LLAMA2_DIR}/forward.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -expand-strided-metadata
            -convert-linalg-to-loops
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-vector-to-llvm
            -llvm-request-c-wrappers
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard -o forward.o
  DEPENDS ${MODEL_LLAMA2_DIR}/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)

add_custom_command(
    OUTPUT subgraph0.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_LLAMA2_DIR}/subgraph0.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-trace-to-llvm="cycle-trace"
            -expand-strided-metadata
            -convert-linalg-to-loops
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-vector-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard -o subgraph0.o
    DEPENDS ${MODEL_LLAMA2_DIR}/subgraph0.mlir
    COMMENT "Building subgraph0.o"
    VERBATIM)

add_library(BuckyballLLAMA2 STATIC forward.o subgraph0.o)
set_target_properties(BuckyballLLAMA2 PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(BuckyballLLAMA2 llama2-model-build)

add_executable(buddy-buckyball-llama2-run ${MODEL_LLAMA2_DIR}/llama-main.cpp)
add_dependencies(buddy-buckyball-llama2-run BuckyballLLAMA2)

target_include_directories(buddy-buckyball-llama2-run PRIVATE ${INTERFACE_DIR} ${INCLUDE_DIR})

set(BUDDY_LLAMA_LIBS BuckyballLLAMA2 ModelTestCRunnerUtils)

target_link_libraries(buddy-buckyball-llama2-run -static ${BUDDY_LLAMA_LIBS})

set(LLAMA2_OUTPUT_DIR ${OUTPUT_BIN_DIR}/src/ModelTest/e2e/models/archs/buckyball/llama2)

add_custom_command(TARGET buddy-buckyball-llama2-run POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${LLAMA2_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:buddy-buckyball-llama2-run>
          ${LLAMA2_OUTPUT_DIR}/buddy-buckyball-llama2-run
  COMMENT "Copying Buckyball llama2 executable to output directory"
  VERBATIM
)

add_custom_target(sync-buckyball-llama2-deps
  COMMAND ${CMAKE_COMMAND} -E make_directory ${LLAMA2_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${LLAMA2_OUTPUT_DIR}/trace/cycle
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_LLAMA2_DIR}/arg0.data
          ${LLAMA2_OUTPUT_DIR}/arg0.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_LLAMA2_DIR}/vocab.txt
          ${LLAMA2_OUTPUT_DIR}/vocab.txt
  DEPENDS
          ${MODEL_LLAMA2_DIR}/arg0.data
          ${MODEL_LLAMA2_DIR}/vocab.txt
  COMMENT "Syncing llama2 runtime deps to output directory"
  VERBATIM
)

add_dependencies(buddy-buckyball-llama2-run sync-buckyball-llama2-deps)
