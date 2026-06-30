set(CMAKE_CXX_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

set(MODEL_MOBILENETV3_DIR ${MODEL_DIR}/MobileNetV3)
set(BUCKYBALL_MOBILENETV3_DIR ${BUCKYBALL_DIR}/MobileNetV3)
set(INTERFACE_DIR ${BUDDY_MLIR_DIR}/frontend/Interfaces)
set(INCLUDE_DIR ${BUCKYBALL_MOBILENETV3_DIR}/include)


add_custom_command(
  OUTPUT  forward.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_MOBILENETV3_DIR}/forward.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
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
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard -o forward.o
          DEPENDS ${MODEL_MOBILENETV3_DIR}/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)

add_custom_command(
  OUTPUT  subgraph0.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_MOBILENETV3_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))"
            > subgraph0_linalg.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_MOBILENETV3_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-tile
            -convert-tile-to-buckyball
            -batchmatmul-optimize
            > subgraph0_loops.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_MOBILENETV3_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-tile
            -convert-tile-to-buckyball
            -lower-buckyball-to-bank-ssa
            -assign-physical-banks
            -lower-bank-ssa-to-intrinsics
            -lower-buckyball
            > subgraph0_buckyball.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_MOBILENETV3_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
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
            -lower-buckyball
            -lower-affine
            -convert-scf-to-cf
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
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_MOBILENETV3_DIR}/subgraph0.mlir
          -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
        ${BUDDY_BINARY_DIR}/buddy-opt
          -eliminate-empty-tensors
          -convert-elementwise-to-linalg
          -convert-tensor-to-linalg
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
          -lower-buckyball
          -lower-affine
          -convert-scf-to-cf
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
        ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard -o subgraph0.o
        DEPENDS ${MODEL_MOBILENETV3_DIR}/subgraph0.mlir
  COMMENT "Building subgraph0.o"
  VERBATIM)


add_library(BuckyballMOBILENETV3 STATIC subgraph0.o forward.o)
set_target_properties(BuckyballMOBILENETV3 PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(BuckyballMOBILENETV3 mobilenetv3-model-build)

add_executable(buddy-buckyball-mobilenetv3-run ${MODEL_MOBILENETV3_DIR}/buddy-mobilenetv3-main.cpp)
add_dependencies(buddy-buckyball-mobilenetv3-run BuckyballMOBILENETV3)

target_include_directories(buddy-buckyball-mobilenetv3-run PRIVATE ${INTERFACE_DIR} ${INCLUDE_DIR})
set(BUDDY_MOBILENETV3_LIBS BuckyballMOBILENETV3 ModelTestCRunnerUtils ModelTestDIP_riscv)
target_link_libraries(buddy-buckyball-mobilenetv3-run -static ${BUDDY_MOBILENETV3_LIBS})

# Define output directory path (mirrors the build directory structure)
set(MOBILENETV3_OUTPUT_DIR ${OUTPUT_BIN_DIR}/src/ModelTest/e2e/models/archs/buckyball/MobileNetV3)

add_custom_command(TARGET buddy-buckyball-mobilenetv3-run POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${MOBILENETV3_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:buddy-buckyball-mobilenetv3-run>
          ${MOBILENETV3_OUTPUT_DIR}/buddy-buckyball-mobilenetv3-run
  COMMENT "Copying Buckyball MobileNetV3 executable to output directory"
  VERBATIM
)

# Sync runtime deps (images/, arg0.data, Labels.txt) directly to OUTPUT directory
add_custom_target(sync-buckyball-mobilenetv3-deps
  COMMAND ${CMAKE_COMMAND} -E make_directory ${MOBILENETV3_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${MOBILENETV3_OUTPUT_DIR}/trace/cycle
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${MODEL_MOBILENETV3_DIR}/images
          ${MOBILENETV3_OUTPUT_DIR}/images
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_MOBILENETV3_DIR}/arg0.data
          ${MOBILENETV3_OUTPUT_DIR}/arg0.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_MOBILENETV3_DIR}/Labels.txt
          ${MOBILENETV3_OUTPUT_DIR}/Labels.txt
  DEPENDS
          ${MODEL_MOBILENETV3_DIR}/arg0.data
          ${MODEL_MOBILENETV3_DIR}/Labels.txt
  COMMENT "Syncing MobileNetV3 runtime deps to output directory"
  VERBATIM
)

add_dependencies(buddy-buckyball-mobilenetv3-run sync-buckyball-mobilenetv3-deps)
