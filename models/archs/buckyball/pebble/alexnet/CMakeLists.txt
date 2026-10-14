set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")
# Model parameters and quantization metadata under OUTPUT_DIR.
set(PAYLOAD_DIR "${OUTPUT_DIR}/alexnet.payload")

file(GLOB_RECURSE DESIGN_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/quant/*"
  "${CMAKE_CURRENT_SOURCE_DIR}/trace/*" "${CMAKE_CURRENT_SOURCE_DIR}/permute/*"
)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/forward.mlir
         ${OUTPUT_DIR}/subgraph0.mlir
         ${OUTPUT_DIR}/subgraph1.mlir
         ${OUTPUT_DIR}/subgraph2.mlir
         ${OUTPUT_DIR}/subgraph3.mlir
         ${OUTPUT_DIR}/subgraph4.mlir
         ${OUTPUT_DIR}/subgraph5.mlir
         ${OUTPUT_DIR}/subgraph6.mlir
         ${OUTPUT_DIR}/subgraph7.mlir
         ${OUTPUT_DIR}/subgraph8.mlir
         ${OUTPUT_DIR}/model.json
         ${OUTPUT_DIR}/input.f32
         ${OUTPUT_DIR}/reference.f32
         ${PAYLOAD_DIR}/params.f32
         ${PAYLOAD_DIR}/weights.bin
         ${PAYLOAD_DIR}/scales.bin
         ${PAYLOAD_DIR}/quant-index.json
  COMMAND ${PYTHON} buddy-pytorch-importer.py
            --repo-root ${REPO_ROOT}
            --weights "IMAGENET1K_V1"
            --output-dir ${OUTPUT_DIR}
            --compiler-build ${COMPILER_BUILD}
            --chip ${CHIP}
  DEPENDS ${MODEL_DIR}/alexnet/configs/importer-param.py
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${MODEL_DIR}/alexnet/prepare.py
          ${MODEL_DIR}/alexnet/images/dog.bmp
          ${DESIGN_FILES}
          ${CMAKE_CURRENT_SOURCE_DIR}/lowering.py
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
set(KERNEL_OBJECTS)
foreach(
  GRAPH
  subgraph0
  subgraph1
  subgraph2
  subgraph3
  subgraph4
  subgraph5
  subgraph6
  subgraph7
  subgraph8
)
  add_custom_command(
    OUTPUT ${GRAPH}.o
    BYPRODUCTS ${OUTPUT_DIR}/${GRAPH}_linalg.mlir
               ${OUTPUT_DIR}/${GRAPH}_tile.mlir
               ${OUTPUT_DIR}/${GRAPH}_buckyball.mlir
               ${OUTPUT_DIR}/${GRAPH}_bank_ssa.mlir
               ${OUTPUT_DIR}/${GRAPH}_llvm.mlir
    COMMAND ${COMPILER_BUILD_pebble}/bin/buddy-opt ${OUTPUT_DIR}/${GRAPH}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
              -o ${OUTPUT_DIR}/${GRAPH}_linalg.mlir
    COMMAND ${COMPILER_BUILD_pebble}/bin/buddy-opt ${OUTPUT_DIR}/${GRAPH}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
            ${COMPILER_BUILD_pebble}/bin/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -one-shot-bufferize=bufferize-function-boundaries
              -convert-linalg-to-tile
              -o ${OUTPUT_DIR}/${GRAPH}_tile.mlir
    COMMAND ${COMPILER_BUILD_pebble}/bin/buddy-opt ${OUTPUT_DIR}/${GRAPH}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
            ${COMPILER_BUILD_pebble}/bin/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -one-shot-bufferize=bufferize-function-boundaries
              -convert-linalg-to-tile
              -convert-tile-to-buckyball
              -o ${OUTPUT_DIR}/${GRAPH}_buckyball.mlir
    COMMAND ${COMPILER_BUILD_pebble}/bin/buddy-opt ${OUTPUT_DIR}/${GRAPH}.mlir
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
              -o ${OUTPUT_DIR}/${GRAPH}_bank_ssa.mlir
    COMMAND ${COMPILER_BUILD_pebble}/bin/buddy-opt ${OUTPUT_DIR}/${GRAPH}.mlir
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
              -llvm-request-c-wrappers
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
              -o ${OUTPUT_DIR}/${GRAPH}_llvm.mlir
    COMMAND ${COMPILER_BUILD_pebble}/bin/buddy-opt ${OUTPUT_DIR}/${GRAPH}.mlir
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
              -llvm-request-c-wrappers
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
            ${COMPILER_BUILD_pebble}/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o ${GRAPH}.o
    COMMAND ${CMAKE_OBJCOPY}
              --redefine-syms=${THREAD_RUNTIME_DIR}/pebble.rename ${GRAPH}.o
    DEPENDS "${OUTPUT_DIR}/${GRAPH}.mlir"
            ${COMPILER_BUILD_pebble}/bin/buddy-opt
            ${COMPILER_BUILD_pebble}/bin/buddy-translate
            ${COMPILER_BUILD_pebble}/bin/buddy-llc
            ${THREAD_RUNTIME_DIR}/pebble.rename
    COMMENT "Building ${GRAPH}.o"
    VERBATIM
  )
  list(APPEND KERNEL_OBJECTS ${GRAPH}.o)
endforeach()
add_library(alexnet-buckyball-pebble-model STATIC EXCLUDE_FROM_ALL forward.o ${KERNEL_OBJECTS})
set_target_properties(alexnet-buckyball-pebble-model PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
set_target_properties(alexnet-buckyball-pebble-model PROPERTIES LINKER_LANGUAGE CXX)

add_executable(alexnet-run EXCLUDE_FROM_ALL ${MODEL_DIR}/alexnet/main.cpp)
set_target_properties(alexnet-run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
add_dependencies(alexnet-run model-loader)
target_include_directories(alexnet-run PRIVATE ${OUTPUT_DIR}
                                               ${BUDDY_SOURCE}/frontend/Interfaces
                                               ${MODEL_LIB_DIR}
)
target_link_libraries(alexnet-run PRIVATE -static
                                          alexnet-buckyball-pebble-model
                                          runtime
                                          scalar_kernels
                                          model_image
)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/alexnet.rax
  BYPRODUCTS ${OUTPUT_DIR}/layout.json
             ${OUTPUT_DIR}/alexnet.rhal.mlir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
  COMMAND ${PYTHON} pack.py
            --repo-root ${REPO_ROOT}
            --output-dir ${OUTPUT_DIR}
            --rax-pack ${COMPILER_BUILD}/bin/rax-pack
            --arguments
            --backend buckyball
  DEPENDS alexnet-run
          ${CMAKE_CURRENT_SOURCE_DIR}/pack.py
          ${OUTPUT_DIR}/model.json
          ${OUTPUT_DIR}/forward.mlir
          ${OUTPUT_DIR}/subgraph0.mlir
          ${OUTPUT_DIR}/subgraph1.mlir
          ${OUTPUT_DIR}/subgraph2.mlir
          ${OUTPUT_DIR}/subgraph3.mlir
          ${OUTPUT_DIR}/subgraph4.mlir
          ${OUTPUT_DIR}/subgraph5.mlir
          ${OUTPUT_DIR}/subgraph6.mlir
          ${OUTPUT_DIR}/subgraph7.mlir
          ${OUTPUT_DIR}/subgraph8.mlir
          ${OUTPUT_DIR}/input.f32
          ${OUTPUT_DIR}/reference.f32
          ${PAYLOAD_DIR}/params.f32
          ${PAYLOAD_DIR}/weights.bin
          ${COMPILER_BUILD}/bin/rax-pack
          ${PAYLOAD_DIR}/scales.bin
          ${PAYLOAD_DIR}/quant-index.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging alexnet model"
  VERBATIM
)
add_custom_target(alexnet-model-build DEPENDS ${OUTPUT_DIR}/alexnet.rax)
