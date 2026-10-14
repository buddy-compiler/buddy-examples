set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")
# Model parameters and quantization metadata under OUTPUT_DIR.
set(PAYLOAD_DIR "${OUTPUT_DIR}/lenet.payload")

#===------------------------------------------------------------------===#
# Track model import dependencies and reconfigure when matching files change.
#===------------------------------------------------------------------===#
file(GLOB_RECURSE DESIGN_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.py
  ${CMAKE_CURRENT_SOURCE_DIR}/trace/*.toml
)

#===------------------------------------------------------------------===#
# Import LeNet into MLIR and export its quantized model data.
#===------------------------------------------------------------------===#
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/forward.mlir
         ${OUTPUT_DIR}/subgraph0.mlir
         ${OUTPUT_DIR}/model.json
         ${PAYLOAD_DIR}/params.f32
         ${PAYLOAD_DIR}/weights.bin
         ${PAYLOAD_DIR}/scales.bin
         ${PAYLOAD_DIR}/quant-index.json
  BYPRODUCTS ${PAYLOAD_DIR}/quant.rhal.mlir
  COMMAND ${PYTHON} buddy-pytorch-importer.py
            --repo-root ${REPO_ROOT}
            --output-dir ${OUTPUT_DIR}
            --compiler-build ${COMPILER_BUILD_pebble}
            --checkpoint ${MODEL_DIR}/lenet/output/lenet-model.pth
  DEPENDS ${MODEL_DIR}/lenet/output/lenet-model.pth
          ${DESIGN_FILES}
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${MODEL_DIR}/lenet/model.py
          ${MODEL_DIR}/lenet/images/8.bmp
          ${COMPILER_BUILD_pebble}/bin/rax-pack
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing LeNet model"
  VERBATIM
)

#===------------------------------------------------------------------===#
# Lower the model entry function to a RISC-V object.
#===------------------------------------------------------------------===#
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
  DEPENDS ${OUTPUT_DIR}/forward.mlir
          ${COMPILER_BUILD_pebble}/bin/buddy-opt
          ${COMPILER_BUILD_pebble}/bin/buddy-translate
          ${COMPILER_BUILD_pebble}/bin/buddy-llc
          ${THREAD_RUNTIME_DIR}/pebble.rename
  COMMENT "Building forward.o"
  VERBATIM
)

#===------------------------------------------------------------------===#
# Lower the compute graph and export intermediate MLIR.
#===------------------------------------------------------------------===#
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
            -one-shot-bufferize=bufferize-function-boundaries
            -convert-linalg-to-tile
            -o ${OUTPUT_DIR}/subgraph0_tile.mlir
  COMMAND ${COMPILER_BUILD_pebble}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
          ${COMPILER_BUILD_pebble}/bin/buddy-opt
            -eliminate-empty-tensors
            -one-shot-bufferize=bufferize-function-boundaries
            -convert-linalg-to-tile
            -convert-tile-to-buckyball
            -o ${OUTPUT_DIR}/subgraph0_buckyball.mlir
  COMMAND ${COMPILER_BUILD_pebble}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
          ${COMPILER_BUILD_pebble}/bin/buddy-opt
            -eliminate-empty-tensors
            -one-shot-bufferize=bufferize-function-boundaries
            -convert-linalg-to-tile
            -convert-tile-to-buckyball
            -lower-buckyball-to-bank-ssa
            -assign-physical-banks
            -llvm-request-c-wrappers
            -o ${OUTPUT_DIR}/subgraph0_bank_ssa.mlir
  COMMAND ${COMPILER_BUILD_pebble}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
          ${COMPILER_BUILD_pebble}/bin/buddy-opt
            -eliminate-empty-tensors
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
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${COMPILER_BUILD_pebble}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
          ${COMPILER_BUILD_pebble}/bin/buddy-opt
            -eliminate-empty-tensors
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
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${COMPILER_BUILD_pebble}/bin/buddy-translate
            --buddy-to-llvmir |
          ${COMPILER_BUILD_pebble}/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o subgraph0.o
  COMMAND ${CMAKE_OBJCOPY}
            --redefine-syms=${THREAD_RUNTIME_DIR}/pebble.rename subgraph0.o
  DEPENDS ${OUTPUT_DIR}/subgraph0.mlir
          ${COMPILER_BUILD_pebble}/bin/buddy-opt
          ${COMPILER_BUILD_pebble}/bin/buddy-translate
          ${COMPILER_BUILD_pebble}/bin/buddy-llc
          ${THREAD_RUNTIME_DIR}/pebble.rename
  COMMENT "Building subgraph0.o"
  VERBATIM
)

# LeNet model code compiled for Pebble; lenet-run links it with application code and runtime.
add_library(lenet-buckyball-pebble-model STATIC EXCLUDE_FROM_ALL forward.o subgraph0.o)
set_target_properties(lenet-buckyball-pebble-model PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
set_target_properties(lenet-buckyball-pebble-model PROPERTIES LINKER_LANGUAGE CXX)

#===------------------------------------------------------------------===#
# Link the model executable.
#===------------------------------------------------------------------===#
add_executable(lenet-run EXCLUDE_FROM_ALL ${MODEL_DIR}/lenet/main.cpp)
set_target_properties(lenet-run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
target_include_directories(lenet-run PRIVATE ${BUDDY_SOURCE}/frontend/Interfaces
                                             ${MODEL_LIB_DIR}
)
target_link_libraries(lenet-run PRIVATE -static
                                        lenet-buckyball-pebble-model
                                        runtime
                                        scalar_kernels
                                        model_image
)

#===------------------------------------------------------------------===#
# Assemble the executable and model data into a RAX file.
#===------------------------------------------------------------------===#
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/lenet.rax
  BYPRODUCTS ${OUTPUT_DIR}/layout.json
             ${OUTPUT_DIR}/lenet.rhal.mlir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
  COMMAND ${PYTHON} pack.py
            --repo-root ${REPO_ROOT}
            --output-dir ${OUTPUT_DIR}
            --rax-pack ${COMPILER_BUILD_pebble}/bin/rax-pack
            --arguments=--image
            --backend buckyball
  DEPENDS lenet-run
          ${OUTPUT_DIR}/model.json
          ${PAYLOAD_DIR}/params.f32
          ${PAYLOAD_DIR}/weights.bin
          ${PAYLOAD_DIR}/scales.bin
          ${PAYLOAD_DIR}/quant-index.json
          ${PAYLOAD_DIR}/quant.rhal.mlir
          ${CMAKE_CURRENT_SOURCE_DIR}/pack.py
          ${COMPILER_BUILD_pebble}/bin/rax-pack
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging LeNet model"
  VERBATIM
)

add_custom_target(lenet-model-build DEPENDS ${OUTPUT_DIR}/lenet.rax)
add_dependencies(lenet-model-build lenet-prepare model-loader)
