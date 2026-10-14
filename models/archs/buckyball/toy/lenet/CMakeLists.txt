set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

#===------------------------------------------------------------------===#
# These are model import dependencies.
# Reconfigure when matching files are added or removed.
#===------------------------------------------------------------------===#
file(GLOB_RECURSE DESIGN_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/quant/*"
  "${CMAKE_CURRENT_SOURCE_DIR}/trace/*" "${CMAKE_CURRENT_SOURCE_DIR}/permute/*"
)

#===------------------------------------------------------------------===#
# Import LeNet into MLIR and export its quantized model data.
#===------------------------------------------------------------------===#
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/forward.mlir
         ${OUTPUT_DIR}/subgraph0.mlir
         ${OUTPUT_DIR}/model.json
         ${OUTPUT_DIR}/lenet.payload/params.f32
         ${OUTPUT_DIR}/lenet.payload/weights.bin
         ${OUTPUT_DIR}/lenet.payload/scales.bin
         ${OUTPUT_DIR}/lenet.payload/quant-index.json
         ${OUTPUT_DIR}/lenet.payload/quant.rhal.mlir
  COMMAND ${PYTHON} buddy-pytorch-importer.py
            --repo-root ${REPO_ROOT}
            --output-dir ${OUTPUT_DIR}
            --compiler-build ${COMPILER_BUILD_toy}
            --checkpoint ${MODEL_DIR}/lenet/output/lenet-model.pth
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${MODEL_DIR}/lenet/output/lenet-model.pth
          ${DESIGN_FILES}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)

#===------------------------------------------------------------------===#
# Lower the model entry function to a RISC-V object.
#===------------------------------------------------------------------===#
add_custom_command(
  OUTPUT forward.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${COMPILER_BUILD_toy}/bin/buddy-opt ${OUTPUT_DIR}/forward.mlir
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
  COMMAND ${COMPILER_BUILD_toy}/bin/buddy-opt ${OUTPUT_DIR}/forward.mlir
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
          ${COMPILER_BUILD_toy}/bin/buddy-translate
            --buddy-to-llvmir |
          ${COMPILER_BUILD_toy}/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o forward.o
  COMMAND ${CMAKE_OBJCOPY}
            --redefine-syms=${THREAD_RUNTIME_DIR}/toy.rename forward.o
  DEPENDS "${OUTPUT_DIR}/forward.mlir"
          ${COMPILER_BUILD_toy}/bin/buddy-opt
          ${COMPILER_BUILD_toy}/bin/buddy-translate
          ${COMPILER_BUILD_toy}/bin/buddy-llc
          ${THREAD_RUNTIME_DIR}/toy.rename
  COMMENT "Building forward.o"
  VERBATIM
)

#===------------------------------------------------------------------===#
# Lower the compute graph and export intermediate MLIR.
#===------------------------------------------------------------------===#
add_custom_command(
  OUTPUT subgraph0.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_bank_ssa.mlir
             ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${COMPILER_BUILD_toy}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_linalg.mlir
  COMMAND ${COMPILER_BUILD_toy}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
          ${COMPILER_BUILD_toy}/bin/buddy-opt
            -eliminate-empty-tensors
            -one-shot-bufferize=bufferize-function-boundaries
            -lower-buckyball-to-bank-ssa
            -assign-physical-banks
            -o ${OUTPUT_DIR}/subgraph0_bank_ssa.mlir
  COMMAND ${COMPILER_BUILD_toy}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
          ${COMPILER_BUILD_toy}/bin/buddy-opt
            -eliminate-empty-tensors
            -one-shot-bufferize=bufferize-function-boundaries
            -lower-buckyball-to-bank-ssa
            -assign-physical-banks
            -llvm-request-c-wrappers
            -convert-trace-to-llvm
            -lower-bank-ssa-to-intrinsics
            -lower-buckyball
            -convert-linalg-to-loops
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${COMPILER_BUILD_toy}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
          ${COMPILER_BUILD_toy}/bin/buddy-opt
            -eliminate-empty-tensors
            -one-shot-bufferize=bufferize-function-boundaries
            -lower-buckyball-to-bank-ssa
            -assign-physical-banks
            -llvm-request-c-wrappers
            -convert-trace-to-llvm
            -lower-bank-ssa-to-intrinsics
            -lower-buckyball
            -convert-linalg-to-loops
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${COMPILER_BUILD_toy}/bin/buddy-translate
            --buddy-to-llvmir |
          ${COMPILER_BUILD_toy}/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o subgraph0.o
  COMMAND ${CMAKE_OBJCOPY}
            --redefine-syms=${THREAD_RUNTIME_DIR}/toy.rename subgraph0.o
  DEPENDS "${OUTPUT_DIR}/subgraph0.mlir"
          ${COMPILER_BUILD_toy}/bin/buddy-opt
          ${COMPILER_BUILD_toy}/bin/buddy-translate
          ${COMPILER_BUILD_toy}/bin/buddy-llc
          ${THREAD_RUNTIME_DIR}/toy.rename
  COMMENT "Building subgraph0.o"
  VERBATIM
)
#===------------------------------------------------------------------===#
# Link the model executable.
#===------------------------------------------------------------------===#
add_library(lenet-buckyball-toy-model STATIC EXCLUDE_FROM_ALL forward.o subgraph0.o)
set_target_properties(lenet-buckyball-toy-model PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
set_target_properties(lenet-buckyball-toy-model PROPERTIES LINKER_LANGUAGE CXX)

add_executable(lenet-run EXCLUDE_FROM_ALL ${MODEL_DIR}/lenet/main.cpp)
set_target_properties(lenet-run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
target_include_directories(lenet-run PRIVATE ${BUDDY_SOURCE}/frontend/Interfaces
                                             ${MODEL_LIB_DIR}
)
target_link_libraries(lenet-run PRIVATE -static
                                        lenet-buckyball-toy-model
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
            --rax-pack ${COMPILER_BUILD_toy}/bin/rax-pack
            --arguments=--image
            --backend buckyball
  DEPENDS lenet-run
          ${CMAKE_CURRENT_SOURCE_DIR}/pack.py
          ${OUTPUT_DIR}/model.json
          ${OUTPUT_DIR}/lenet.payload/params.f32
          ${OUTPUT_DIR}/lenet.payload/weights.bin
          ${COMPILER_BUILD_toy}/bin/rax-pack
          ${OUTPUT_DIR}/lenet.payload/scales.bin
          ${OUTPUT_DIR}/lenet.payload/quant-index.json
          ${OUTPUT_DIR}/lenet.payload/quant.rhal.mlir
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging lenet model"
  VERBATIM
)
add_custom_target(lenet-model-build DEPENDS ${OUTPUT_DIR}/lenet.rax)
add_dependencies(lenet-model-build lenet-prepare model-loader)
