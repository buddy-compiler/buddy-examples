set(CMAKE_CXX_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

set(MODEL_YOLO26_DIR ${MODEL_DIR}/YOLO26)
set(INTERFACE_DIR ${BUDDY_MLIR_DIR}/frontend/Interfaces)

add_custom_command(
  OUTPUT forward.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_YOLO26_DIR}/forward.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -expand-strided-metadata
            -convert-linalg-to-loops
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard -o forward.o
  DEPENDS ${MODEL_YOLO26_DIR}/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph0.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_YOLO26_DIR}/subgraph0.mlir
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
  DEPENDS ${MODEL_YOLO26_DIR}/subgraph0.mlir
  COMMENT "Building subgraph0.o"
  VERBATIM)

add_library(BuckyballYOLO26 STATIC forward.o subgraph0.o)
set_target_properties(BuckyballYOLO26 PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(BuckyballYOLO26 yolo26-model-build)

add_executable(buddy-buckyball-yolo26-run ${MODEL_YOLO26_DIR}/buddy-yolo26n-main.cpp)
add_dependencies(buddy-buckyball-yolo26-run BuckyballYOLO26)

target_compile_definitions(buddy-buckyball-yolo26-run PRIVATE
  YOLO26_EXAMPLE_PATH="${MODEL_YOLO26_DIR}/"
  YOLO26_EXAMPLE_BUILD_PATH="${CMAKE_CURRENT_BINARY_DIR}/"
)
target_include_directories(buddy-buckyball-yolo26-run PRIVATE ${INTERFACE_DIR})
set(BUDDY_YOLO26_LIBS BuckyballYOLO26 ModelTestCRunnerUtils ModelTestDIP_riscv)
target_link_libraries(buddy-buckyball-yolo26-run -static ${BUDDY_YOLO26_LIBS})

# Define output directory path (mirrors the build directory structure)
set(YOLO26_OUTPUT_DIR ${OUTPUT_BIN_DIR}/src/ModelTest/e2e/models/archs/buckyball/YOLO26)

add_custom_command(TARGET buddy-buckyball-yolo26-run POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${YOLO26_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:buddy-buckyball-yolo26-run>
          ${YOLO26_OUTPUT_DIR}/buddy-buckyball-yolo26-run
  COMMENT "Copying Buckyball YOLO26 executable to output directory"
  VERBATIM
)

# Sync runtime deps (images/, arg0.data, labels.txt) directly to OUTPUT directory
add_custom_target(sync-buckyball-yolo26-deps
  COMMAND ${CMAKE_COMMAND} -E make_directory ${YOLO26_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${YOLO26_OUTPUT_DIR}/trace/cycle
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${MODEL_YOLO26_DIR}/images
          ${YOLO26_OUTPUT_DIR}/images
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_YOLO26_DIR}/arg0.data
          ${YOLO26_OUTPUT_DIR}/arg0.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_YOLO26_DIR}/labels.txt
          ${YOLO26_OUTPUT_DIR}/labels.txt
  DEPENDS
          ${MODEL_YOLO26_DIR}/arg0.data
          ${MODEL_YOLO26_DIR}/labels.txt
  COMMENT "Syncing YOLO26 runtime deps to output directory"
  VERBATIM
)

add_dependencies(buddy-buckyball-yolo26-run sync-buckyball-yolo26-deps)
