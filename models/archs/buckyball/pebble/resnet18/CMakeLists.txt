set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")
# Model parameters and quantization metadata under OUTPUT_DIR.
set(PAYLOAD_DIR "${OUTPUT_DIR}/resnet18.payload")

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
            --weights "DEFAULT"
            --output-dir ${OUTPUT_DIR}
            --compiler-build ${COMPILER_BUILD}
  DEPENDS ${MODEL_DIR}/resnet18/configs/importer-param.py
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${DESIGN_FILES}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)

add_custom_command(
  OUTPUT forward.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${COMPILER_BUILD_pebble}/bin/buddy-opt ${OUTPUT_DIR}/forward.mlir
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
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${COMPILER_BUILD_pebble}/bin/buddy-opt ${OUTPUT_DIR}/forward.mlir
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
            -expand-strided-metadata
            -convert-linalg-to-loops
            -lower-buckyball
            -canonicalize
            -cse
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
            -expand-strided-metadata
            -convert-linalg-to-loops
            -lower-buckyball
            -canonicalize
            -cse
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
add_library(resnet18-buckyball-pebble-model STATIC EXCLUDE_FROM_ALL forward.o subgraph0.o)
set_target_properties(resnet18-buckyball-pebble-model PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
set_target_properties(resnet18-buckyball-pebble-model PROPERTIES LINKER_LANGUAGE CXX)

add_executable(resnet18-run EXCLUDE_FROM_ALL ${MODEL_DIR}/resnet18/main.cpp)
set_target_properties(resnet18-run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
add_dependencies(resnet18-run model-loader)
target_include_directories(resnet18-run PRIVATE ${OUTPUT_DIR}
                                                ${BUDDY_SOURCE}/frontend/Interfaces
                                                ${MODEL_LIB_DIR}
)
target_link_libraries(resnet18-run PRIVATE -static
                                           resnet18-buckyball-pebble-model
                                           runtime
                                           scalar_kernels
                                           model_image
)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/resnet18.rax
  BYPRODUCTS ${OUTPUT_DIR}/layout.json
             ${OUTPUT_DIR}/resnet18.rhal.mlir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
  COMMAND ${PYTHON} pack.py
            --repo-root ${REPO_ROOT}
            --output-dir ${OUTPUT_DIR}
            --rax-pack ${COMPILER_BUILD}/bin/rax-pack
            --arguments
            --backend buckyball
  DEPENDS resnet18-run
          ${CMAKE_CURRENT_SOURCE_DIR}/pack.py
          ${OUTPUT_DIR}/forward.mlir
          ${OUTPUT_DIR}/subgraph0.mlir
          ${OUTPUT_DIR}/model.json
          ${PAYLOAD_DIR}/params.f32
          ${PAYLOAD_DIR}/weights.bin
          ${COMPILER_BUILD}/bin/rax-pack
          ${PAYLOAD_DIR}/scales.bin
          ${PAYLOAD_DIR}/quant-index.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging resnet18 model"
  VERBATIM
)
add_custom_target(resnet18-model-build DEPENDS ${OUTPUT_DIR}/resnet18.rax)
