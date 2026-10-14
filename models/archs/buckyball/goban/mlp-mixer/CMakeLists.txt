set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")
# Model parameters and quantization metadata under OUTPUT_DIR.
set(PAYLOAD_DIR "${OUTPUT_DIR}/mixer.payload")
# Model parameters and quantization metadata under OUTPUT_DIR.
set(PAYLOAD_DIR "${PAYLOAD_DIR}")

set(KERNEL_MLIR)
set(KERNEL_TASKS)
foreach(INDEX RANGE 0 49)
  list(APPEND KERNEL_MLIR ${OUTPUT_DIR}/subgraph${INDEX}.mlir ${OUTPUT_DIR}/linear${INDEX}.mlir)
  list(APPEND KERNEL_TASKS ${OUTPUT_DIR}/linear${INDEX}-task.c)
endforeach()
file(GLOB_RECURSE DESIGN_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/quant/*"
  "${CMAKE_CURRENT_SOURCE_DIR}/trace/*" "${CMAKE_CURRENT_SOURCE_DIR}/permute/*"
)
add_custom_command(
  OUTPUT ${KERNEL_MLIR}
         ${KERNEL_TASKS}
         ${OUTPUT_DIR}/mixer-parameters.h
         ${OUTPUT_DIR}/model.json
         ${OUTPUT_DIR}/input.f32
         ${OUTPUT_DIR}/reference.f32
         ${PAYLOAD_DIR}/params.f32
         ${PAYLOAD_DIR}/weights.bin
         ${PAYLOAD_DIR}/scales.bin
         ${PAYLOAD_DIR}/quant-index.json
  COMMAND ${PYTHON} buddy-pytorch-importer.py
            --repo-root ${REPO_ROOT}
            --checkpoint "timm/mixer_b16_224.goog_in21k_ft_in1k"
            --revision "0006300d604941f96f16cb7d229137363e1b2ac6"
            --output-dir ${OUTPUT_DIR}
            --compiler-build ${COMPILER_BUILD}
            --chip-pb ${CHIP_PB}
            --chip ${CHIP}
  DEPENDS ${MODEL_DIR}/mlp-mixer/configs/importer-param.py
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${MODEL_DIR}/mlp-mixer/prepare.py
          ${MODEL_DIR}/mlp-mixer/../alexnet/images/dog.bmp
          ${DESIGN_FILES}
          ${CMAKE_CURRENT_SOURCE_DIR}/lowering.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)
set(KERNEL_OBJECTS)
foreach(INDEX RANGE 0 49)
  add_custom_command(
    OUTPUT linear${INDEX}.o
    BYPRODUCTS ${OUTPUT_DIR}/linear${INDEX}_llvm.mlir
    COMMAND ${COMPILER_BUILD_goban}/bin/buddy-opt ${OUTPUT_DIR}/linear${INDEX}.mlir
              -buffer-deallocation-pipeline
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
              -o ${OUTPUT_DIR}/linear${INDEX}_llvm.mlir
    COMMAND ${COMPILER_BUILD_goban}/bin/buddy-opt ${OUTPUT_DIR}/linear${INDEX}.mlir
              -buffer-deallocation-pipeline
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
            ${COMPILER_BUILD_goban}/bin/buddy-translate
              --buddy-to-llvmir |
            ${COMPILER_BUILD_goban}/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o linear${INDEX}.o
    COMMAND ${CMAKE_OBJCOPY}
              --redefine-syms=${THREAD_RUNTIME_DIR}/goban.rename
              --redefine-sym malloc=workspace_alloc
              --redefine-sym free=workspace_free linear${INDEX}.o
    DEPENDS "${OUTPUT_DIR}/linear${INDEX}.mlir"
            ${COMPILER_BUILD_goban}/bin/buddy-opt
            ${COMPILER_BUILD_goban}/bin/buddy-translate
            ${COMPILER_BUILD_goban}/bin/buddy-llc
            ${THREAD_RUNTIME_DIR}/goban.rename
    COMMENT "Building linear${INDEX}.o"
    VERBATIM
  )

  add_custom_command(
    OUTPUT subgraph${INDEX}.o
    BYPRODUCTS ${OUTPUT_DIR}/subgraph${INDEX}_linalg.mlir
               ${OUTPUT_DIR}/subgraph${INDEX}_tile.mlir
               ${OUTPUT_DIR}/subgraph${INDEX}_buckyball.mlir
               ${OUTPUT_DIR}/subgraph${INDEX}_bank_ssa.mlir
               ${OUTPUT_DIR}/subgraph${INDEX}_llvm.mlir
    COMMAND ${COMPILER_BUILD_goban}/bin/buddy-opt ${OUTPUT_DIR}/subgraph${INDEX}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
              -o ${OUTPUT_DIR}/subgraph${INDEX}_linalg.mlir
    COMMAND ${COMPILER_BUILD_goban}/bin/buddy-opt ${OUTPUT_DIR}/subgraph${INDEX}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
            ${COMPILER_BUILD_goban}/bin/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -one-shot-bufferize=bufferize-function-boundaries
              -buffer-deallocation-pipeline
              -convert-linalg-to-tile=quantized-only
              -o ${OUTPUT_DIR}/subgraph${INDEX}_tile.mlir
    COMMAND ${COMPILER_BUILD_goban}/bin/buddy-opt ${OUTPUT_DIR}/subgraph${INDEX}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
            ${COMPILER_BUILD_goban}/bin/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -one-shot-bufferize=bufferize-function-boundaries
              -buffer-deallocation-pipeline
              -convert-linalg-to-tile=quantized-only
              -convert-tile-to-buckyball
              -o ${OUTPUT_DIR}/subgraph${INDEX}_buckyball.mlir
    COMMAND ${COMPILER_BUILD_goban}/bin/buddy-opt ${OUTPUT_DIR}/subgraph${INDEX}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
            ${COMPILER_BUILD_goban}/bin/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -one-shot-bufferize=bufferize-function-boundaries
              -buffer-deallocation-pipeline
              -convert-linalg-to-tile=quantized-only
              -convert-tile-to-buckyball
              -lower-buckyball-to-bank-ssa
              -assign-physical-banks
              -o ${OUTPUT_DIR}/subgraph${INDEX}_bank_ssa.mlir
    COMMAND ${COMPILER_BUILD_goban}/bin/buddy-opt ${OUTPUT_DIR}/subgraph${INDEX}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
            ${COMPILER_BUILD_goban}/bin/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -one-shot-bufferize=bufferize-function-boundaries
              -buffer-deallocation-pipeline
              -convert-linalg-to-tile=quantized-only
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
              -o ${OUTPUT_DIR}/subgraph${INDEX}_llvm.mlir
    COMMAND ${COMPILER_BUILD_goban}/bin/buddy-opt ${OUTPUT_DIR}/subgraph${INDEX}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
            ${COMPILER_BUILD_goban}/bin/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -one-shot-bufferize=bufferize-function-boundaries
              -buffer-deallocation-pipeline
              -convert-linalg-to-tile=quantized-only
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
            ${COMPILER_BUILD_goban}/bin/buddy-translate
              --buddy-to-llvmir |
            ${COMPILER_BUILD_goban}/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o subgraph${INDEX}.o
    COMMAND ${CMAKE_OBJCOPY}
              --redefine-syms=${THREAD_RUNTIME_DIR}/goban.rename
              --redefine-sym malloc=workspace_alloc
              --redefine-sym free=workspace_free subgraph${INDEX}.o
    DEPENDS "${OUTPUT_DIR}/subgraph${INDEX}.mlir"
            ${COMPILER_BUILD_goban}/bin/buddy-opt
            ${COMPILER_BUILD_goban}/bin/buddy-translate
            ${COMPILER_BUILD_goban}/bin/buddy-llc
            ${THREAD_RUNTIME_DIR}/goban.rename
    COMMENT "Building subgraph${INDEX}.o"
    VERBATIM
  )
  list(APPEND KERNEL_OBJECTS linear${INDEX}.o subgraph${INDEX}.o ${OUTPUT_DIR}/linear${INDEX}-task.c)
endforeach()
add_library(mlp-mixer-buckyball-goban-model STATIC EXCLUDE_FROM_ALL ${KERNEL_OBJECTS})
set_target_properties(mlp-mixer-buckyball-goban-model PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
set_target_properties(mlp-mixer-buckyball-goban-model PROPERTIES LINKER_LANGUAGE CXX)

add_executable(mixer-run EXCLUDE_FROM_ALL ${MODEL_DIR}/mlp-mixer/main.cpp)
set_target_properties(mixer-run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
add_dependencies(mixer-run model-loader)
target_include_directories(mixer-run PRIVATE ${OUTPUT_DIR}
                                             ${BUDDY_SOURCE}/frontend/Interfaces
                                             ${BUDDY_SOURCE}/llvm/mlir/include
                                             ${MODEL_LIB_DIR}
)
target_include_directories(mixer-run PRIVATE ${RUNTIME_PARAMS})
target_link_libraries(mixer-run PRIVATE -static
                                        mlp-mixer-buckyball-goban-model
                                        runtime
                                        scalar_kernels
)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/mlp-mixer.rax
  BYPRODUCTS ${OUTPUT_DIR}/layout.json
             ${OUTPUT_DIR}/mlp-mixer.rhal.mlir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
  COMMAND ${PYTHON} pack.py
            --repo-root ${REPO_ROOT}
            --output-dir ${OUTPUT_DIR}
            --rax-pack ${COMPILER_BUILD}/bin/rax-pack
            --backend buckyball
  DEPENDS mixer-run
          ${CMAKE_CURRENT_SOURCE_DIR}/pack.py
          ${OUTPUT_DIR}/model.json
          ${KERNEL_MLIR}
          ${KERNEL_TASKS}
          ${OUTPUT_DIR}/mixer-parameters.h
          ${OUTPUT_DIR}/input.f32
          ${OUTPUT_DIR}/reference.f32
          ${PAYLOAD_DIR}/params.f32
          ${PAYLOAD_DIR}/weights.bin
          ${COMPILER_BUILD}/bin/rax-pack
          ${PAYLOAD_DIR}/scales.bin
          ${PAYLOAD_DIR}/quant-index.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging mlp-mixer model"
  VERBATIM
)
add_custom_target(mlp-mixer-model-build DEPENDS ${OUTPUT_DIR}/mlp-mixer.rax)
