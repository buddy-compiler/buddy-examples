set(CMAKE_CXX_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

set(MODEL_WHISPER_DIR ${MODEL_DIR}/Whisper)
set(MODEL_WHISPER_BUILD_DIR ${CMAKE_BINARY_DIR}/workloads/src/ModelTest/e2e/models/models/Whisper)
set(BUCKYBALL_WHISPER_DIR ${BUCKYBALL_DIR}/Whisper)
set(INTERFACE_DIR ${BUDDY_MLIR_DIR}/frontend/Interfaces)

add_custom_command(
  OUTPUT forward.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_WHISPER_BUILD_DIR}/forward.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg)" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -buffer-deallocation-simplification
            -convert-linalg-to-loops
            -eliminate-empty-tensors
            -llvm-request-c-wrappers
            -convert-math-to-llvm
            -convert-math-to-libm
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-arith-to-llvm
            -expand-strided-metadata
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xbbpebble,+D -float-abi=hard -o forward.o
  DEPENDS ${MODEL_WHISPER_BUILD_DIR}/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph0.o
  BYPRODUCTS
         subgraph0_linalg.mlir
         subgraph0_loops.mlir
         subgraph0_buckyball.mlir
         subgraph0.ll
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_WHISPER_BUILD_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
          > subgraph0_linalg.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt subgraph0_linalg.mlir
            -convert-linalg-to-tile
            -convert-tile-to-buckyball
            -batchmatmul-optimize
            -o subgraph0_loops.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt subgraph0_loops.mlir
            -lower-buckyball-to-bank-ssa
            -assign-physical-banks
            -lower-bank-ssa-to-intrinsics
            -lower-buckyball
            -o subgraph0_buckyball.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt subgraph0_buckyball.mlir
            -llvm-request-c-wrappers
            -convert-trace-to-llvm="cycle-trace"
            -expand-strided-metadata
            -convert-linalg-to-loops
            -lower-affine
            -convert-scf-to-cf=allow-pattern-rollback=false
            -convert-cf-to-llvm
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir
          > subgraph0.ll
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt subgraph0_buckyball.mlir
            -llvm-request-c-wrappers
            -convert-trace-to-llvm="cycle-trace"
            -expand-strided-metadata
            -convert-linalg-to-loops
            -lower-affine
            -convert-scf-to-cf=allow-pattern-rollback=false
            -convert-cf-to-llvm
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xbbpebble,+D -float-abi=hard -o subgraph0.o
  DEPENDS ${MODEL_WHISPER_BUILD_DIR}/subgraph0.mlir
  COMMENT "Building subgraph0.o"
  VERBATIM)

add_library(BuckyballWHISPER STATIC subgraph0.o forward.o)
set_target_properties(BuckyballWHISPER PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(BuckyballWHISPER whisper-model-build)

add_executable(buddy-buckyball-whisper-run ${MODEL_WHISPER_DIR}/whisper-main.cpp)
add_dependencies(buddy-buckyball-whisper-run BuckyballWHISPER)
target_include_directories(buddy-buckyball-whisper-run PRIVATE ${INTERFACE_DIR})
target_link_libraries(buddy-buckyball-whisper-run -static
  BuckyballWHISPER
  ${MODELTEST_DAP_EXTEND_RISCV_LIB}
  ${MODELTEST_DAP_RISCV_LIB}
  ${MODELTEST_CRUNNER_UTILS_LIB}
)

set(WHISPER_OUTPUT_DIR ${OUTPUT_BIN_DIR}/src/ModelTest/e2e/models/archs/buckyball/Whisper)

add_custom_command(TARGET buddy-buckyball-whisper-run POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${WHISPER_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:buddy-buckyball-whisper-run>
          ${WHISPER_OUTPUT_DIR}/buddy-buckyball-whisper-run
  COMMENT "Copying Buckyball Whisper executable to output directory"
  VERBATIM
)

add_custom_target(sync-buckyball-whisper-deps
  COMMAND ${CMAKE_COMMAND} -E make_directory ${WHISPER_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${WHISPER_OUTPUT_DIR}/trace/cycle
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_WHISPER_BUILD_DIR}/arg0.data
          ${WHISPER_OUTPUT_DIR}/arg0.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_WHISPER_DIR}/audio.wav
          ${WHISPER_OUTPUT_DIR}/audio.wav
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_WHISPER_DIR}/vocab.txt
          ${WHISPER_OUTPUT_DIR}/vocab.txt
  DEPENDS
          ${MODEL_WHISPER_BUILD_DIR}/arg0.data
          ${MODEL_WHISPER_DIR}/audio.wav
          ${MODEL_WHISPER_DIR}/vocab.txt
  COMMENT "Syncing Whisper runtime deps to output directory"
  VERBATIM
)

add_dependencies(buddy-buckyball-whisper-run sync-buckyball-whisper-deps)
