set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")
# Model parameters and quantization metadata under OUTPUT_DIR.
set(PAYLOAD_DIR "${OUTPUT_DIR}/yolo26.payload")

set(
  CALIBRATION_IMAGES
  ${MODEL_DIR}/resnet18/images/dog.png
  ${MODEL_DIR}/resnet18/images/ice-cream.png
  ${MODEL_DIR}/resnet18/images/traffic-light.png
  ${MODEL_DIR}/resnet18/images/kite.png
  ${MODEL_DIR}/resnet18/images/curtain.png
)
file(GLOB_RECURSE DESIGN_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/quant/*"
  "${CMAKE_CURRENT_SOURCE_DIR}/trace/*" "${CMAKE_CURRENT_SOURCE_DIR}/permute/*"
)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/forward.mlir
         ${OUTPUT_DIR}/subgraph0.mlir
         ${OUTPUT_DIR}/model.json
         ${PAYLOAD_DIR}/params.f32
         ${PAYLOAD_DIR}/weights.bin
         ${PAYLOAD_DIR}/scales.bin
         ${PAYLOAD_DIR}/quant-index.json
  COMMAND ${PYTHON} buddy-pytorch-importer.py
            --repo-root ${REPO_ROOT}
            --img-size 640
            --output-dir ${OUTPUT_DIR}
            --compiler-build ${COMPILER_BUILD}
            --checkpoint ${MODEL_DIR}/yolo26n/output/yolo26n.pt
            --calibration-images ${CALIBRATION_IMAGES}
  DEPENDS ${MODEL_DIR}/yolo26n/configs/importer-param.py
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${MODEL_DIR}/yolo26n/output/yolo26n.pt
          ${DESIGN_FILES}
          ${CALIBRATION_IMAGES}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)

add_custom_command(
  OUTPUT forward.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${COMPILER_BUILD_pebble}/bin/buddy-opt ${OUTPUT_DIR}/forward.mlir
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
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
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${COMPILER_BUILD_pebble}/bin/buddy-opt ${OUTPUT_DIR}/forward.mlir
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
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
          ${COMPILER_BUILD_pebble}/bin/buddy-translate
            --buddy-to-llvmir |
          ${COMPILER_BUILD_pebble}/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o forward.o
  COMMAND ${CMAKE_OBJCOPY}
            --redefine-syms=${THREAD_RUNTIME_DIR}/pebble.rename forward.o
  DEPENDS "${OUTPUT_DIR}/forward.mlir"
          ${COMPILER_BUILD_pebble}/bin/buddy-opt
          ${COMPILER_BUILD_pebble}/bin/buddy-translate
          ${COMPILER_BUILD_pebble}/bin/buddy-llc
          ${THREAD_RUNTIME_DIR}/pebble.rename
  COMMENT "Building forward.o"
  VERBATIM
)

add_custom_command(
  OUTPUT subgraph0.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_tile.mlir
             ${OUTPUT_DIR}/subgraph0_buckyball.mlir
             ${OUTPUT_DIR}/subgraph0_bank_ssa.mlir
             ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${COMPILER_BUILD_pebble}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_linalg.mlir
  COMMAND ${COMPILER_BUILD_pebble}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
          ${COMPILER_BUILD_pebble}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
            -convert-linalg-to-tile
            -o ${OUTPUT_DIR}/subgraph0_tile.mlir
  COMMAND ${COMPILER_BUILD_pebble}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
          ${COMPILER_BUILD_pebble}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
            -convert-linalg-to-tile
            -convert-tile-to-buckyball
            -o ${OUTPUT_DIR}/subgraph0_buckyball.mlir
  COMMAND ${COMPILER_BUILD_pebble}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
          ${COMPILER_BUILD_pebble}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
            -convert-linalg-to-tile
            -convert-tile-to-buckyball
            -lower-buckyball-to-bank-ssa
            -assign-physical-banks
            -o ${OUTPUT_DIR}/subgraph0_bank_ssa.mlir
  COMMAND ${COMPILER_BUILD_pebble}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
          ${COMPILER_BUILD_pebble}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
            -convert-linalg-to-tile
            -convert-tile-to-buckyball
            -lower-buckyball-to-bank-ssa
            -assign-physical-banks
            -llvm-request-c-wrappers |
          ${COMPILER_BUILD_pebble}/bin/buddy-opt
            -lower-bank-ssa-to-intrinsics
            -convert-trace-to-llvm
            -lower-buckyball
            -canonicalize
            -cse
            --mlir-disable-threading
            -llvm-request-c-wrappers
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
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${COMPILER_BUILD_pebble}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
          ${COMPILER_BUILD_pebble}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
            -convert-linalg-to-tile
            -convert-tile-to-buckyball
            -lower-buckyball-to-bank-ssa
            -assign-physical-banks
            -llvm-request-c-wrappers |
          ${COMPILER_BUILD_pebble}/bin/buddy-opt
            -lower-bank-ssa-to-intrinsics
            -convert-trace-to-llvm
            -lower-buckyball
            -canonicalize
            -cse
            --mlir-disable-threading
            -llvm-request-c-wrappers
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
          ${COMPILER_BUILD_pebble}/bin/buddy-translate
            --buddy-to-llvmir |
          ${COMPILER_BUILD_pebble}/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o subgraph0.o
  COMMAND ${CMAKE_OBJCOPY}
            --redefine-syms=${THREAD_RUNTIME_DIR}/pebble.rename subgraph0.o
  DEPENDS ${COMPILER_BUILD_pebble}/bin/buddy-opt
          ${COMPILER_BUILD_pebble}/bin/buddy-translate
          "${OUTPUT_DIR}/subgraph0.mlir"
          ${COMPILER_BUILD_pebble}/bin/buddy-llc
          ${THREAD_RUNTIME_DIR}/pebble.rename
  COMMENT "Building subgraph0.o"
  VERBATIM
)
add_library(yolo26n-buckyball-pebble-model STATIC EXCLUDE_FROM_ALL forward.o subgraph0.o)
set_target_properties(yolo26n-buckyball-pebble-model PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
set_target_properties(yolo26n-buckyball-pebble-model PROPERTIES LINKER_LANGUAGE CXX)

add_executable(yolo26-run EXCLUDE_FROM_ALL ${MODEL_DIR}/yolo26n/main.cpp)
set_target_properties(yolo26-run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
add_dependencies(yolo26-run model-loader)
target_include_directories(yolo26-run PRIVATE ${OUTPUT_DIR}
                                              ${BUDDY_SOURCE}/frontend/Interfaces
                                              ${MODEL_LIB_DIR}
)
target_link_libraries(yolo26-run PRIVATE -static
                                         yolo26n-buckyball-pebble-model
                                         runtime
                                         scalar_kernels
                                         model_image
)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/yolo26.rax
  BYPRODUCTS ${OUTPUT_DIR}/layout.json
             ${OUTPUT_DIR}/yolo26.rhal.mlir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
  COMMAND ${PYTHON} pack.py
            --repo-root ${REPO_ROOT}
            --output-dir ${OUTPUT_DIR}
            --rax-pack ${COMPILER_BUILD}/bin/rax-pack
            --arguments
            --backend buckyball
  DEPENDS yolo26-run
          ${CMAKE_CURRENT_SOURCE_DIR}/pack.py
          ${MODEL_DIR}/yolo26n/output/yolo26n.pt
          ${OUTPUT_DIR}/forward.mlir
          ${OUTPUT_DIR}/subgraph0.mlir
          ${OUTPUT_DIR}/model.json
          ${PAYLOAD_DIR}/params.f32
          ${PAYLOAD_DIR}/weights.bin
          ${COMPILER_BUILD}/bin/rax-pack
          ${PAYLOAD_DIR}/scales.bin
          ${PAYLOAD_DIR}/quant-index.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging yolo26n model"
  VERBATIM
)
add_custom_target(yolo26n-model-build DEPENDS ${OUTPUT_DIR}/yolo26.rax)

add_dependencies(yolo26n-model-build yolo26n-prepare)
