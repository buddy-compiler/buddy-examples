set(CMAKE_CXX_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

set(MODEL_LENET_DIR ${MODEL_DIR}/LeNet)
set(BUCKYBALL_LENET_DIR ${BUCKYBALL_DIR}/LeNet)
set(INTERFACE_DIR ${BUDDY_MLIR_DIR}/frontend/Interfaces)
set(INCLUDE_DIR ${MODEL_LENET_DIR}/include)


add_custom_command(
  OUTPUT  forward.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_LENET_DIR}/forward.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -buffer-deallocation-simplification
            -convert-linalg-to-loops
            -eliminate-empty-tensors
            -llvm-request-c-wrappers
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-scf-to-cf
            -convert-arith-to-llvm
            -expand-strided-metadata
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard -o forward.o
          DEPENDS ${MODEL_LENET_DIR}/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)

add_custom_command(
  OUTPUT  subgraph0.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_LENET_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))"
            > subgraph0_linalg.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_LENET_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-tile
            -convert-tile-to-buckyball
            -batchmatmul-optimize
            > subgraph0_loops.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_LENET_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-tile
            -convert-tile-to-buckyball
            -lower-buckyball-to-bank-ssa
            -assign-physical-banks
            -lower-bank-ssa-to-intrinsics
            -lower-buckyball
            > subgraph0_buckyball.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_LENET_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-tile
            -convert-tile-to-buckyball
            -lower-buckyball-to-bank-ssa
            -assign-physical-banks
            -lower-bank-ssa-to-intrinsics
            -convert-trace-to-llvm="cycle-trace"
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -lower-buckyball
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir
          > subgraph0.ll
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_LENET_DIR}/subgraph0.mlir
          -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
        ${BUDDY_BINARY_DIR}/buddy-opt
          -eliminate-empty-tensors
          -one-shot-bufferize="bufferize-function-boundaries"
          -convert-linalg-to-tile
          -convert-tile-to-buckyball
          -lower-buckyball-to-bank-ssa
          -assign-physical-banks
          -llvm-request-c-wrappers
          -lower-bank-ssa-to-intrinsics
          -convert-trace-to-llvm="cycle-trace"
          -expand-strided-metadata
          -convert-linalg-to-loops
          -convert-scf-to-cf
          -lower-buckyball
          -ownership-based-buffer-deallocation
          -buffer-deallocation-simplification
          -bufferization-lower-deallocations
          -convert-arith-to-llvm
          -convert-func-to-llvm
          -finalize-memref-to-llvm
          -reconcile-unrealized-casts |
        ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
        ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard -o subgraph0.o
        DEPENDS ${MODEL_LENET_DIR}/subgraph0.mlir
  COMMENT "Building subgraph0.o"
  VERBATIM)

add_library(BuckyballLENET STATIC subgraph0.o forward.o)
set_target_properties(BuckyballLENET PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(BuckyballLENET lenet-model-build)

add_executable(buddy-buckyball-lenet-run ${MODEL_LENET_DIR}/buddy-lenet-main.cpp)
add_dependencies(buddy-buckyball-lenet-run BuckyballLENET)
target_include_directories(buddy-buckyball-lenet-run PRIVATE ${INTERFACE_DIR} ${INCLUDE_DIR})
target_link_libraries(buddy-buckyball-lenet-run -static BuckyballLENET ${MODELTEST_CRUNNER_UTILS_LIB})

# Define output directory path (mirrors the build directory structure)
set(LENET_OUTPUT_DIR ${OUTPUT_BIN_DIR}/src/ModelTest/e2e/models/archs/buckyball/LeNet)

add_custom_command(TARGET buddy-buckyball-lenet-run POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${LENET_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:buddy-buckyball-lenet-run>
          ${LENET_OUTPUT_DIR}/buddy-buckyball-lenet-run
  COMMENT "Copying Buckyball LeNet executable to output directory"
  VERBATIM
)

add_custom_target(buddy-buckyball-lenet-all-run
  DEPENDS
    buddy-buckyball-lenet-run
)

# Sync runtime deps (images/, arg0.data) directly to OUTPUT directory
add_custom_target(sync-buckyball-lenet-deps
  COMMAND ${CMAKE_COMMAND} -E make_directory ${LENET_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${LENET_OUTPUT_DIR}/trace/cycle
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${MODEL_LENET_DIR}/images
          ${LENET_OUTPUT_DIR}/images
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_LENET_DIR}/arg0.data
          ${LENET_OUTPUT_DIR}/arg0.data
  DEPENDS ${MODEL_LENET_DIR}/arg0.data
  COMMENT "Syncing LeNet runtime deps to output directory"
  VERBATIM
)

add_dependencies(buddy-buckyball-lenet-run sync-buckyball-lenet-deps)
