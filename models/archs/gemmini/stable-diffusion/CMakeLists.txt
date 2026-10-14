set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

add_custom_command(
  OUTPUT ${OUTPUT_DIR}/arg0_text_encoder.data
         ${OUTPUT_DIR}/arg1_text_encoder.data
         ${OUTPUT_DIR}/arg0_unet.data
         ${OUTPUT_DIR}/arg0_vae.data
         ${OUTPUT_DIR}/forward_text_encoder.mlir
         ${OUTPUT_DIR}/subgraph0_text_encoder.mlir
         ${OUTPUT_DIR}/forward_unet.mlir
         ${OUTPUT_DIR}/subgraph0_unet.mlir
         ${OUTPUT_DIR}/forward_vae.mlir
         ${OUTPUT_DIR}/subgraph0_vae.mlir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
  COMMAND ${PYTHON} buddy-pytorch-importer.py
            --repo-root ${REPO_ROOT}
            --output-dir ${OUTPUT_DIR}
            --compiler-build ${COMPILER_BUILD}
            --checkpoint "borno1/stable_diffusion_2_base"
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing stable-diffusion for gemmini"
  VERBATIM
)

add_custom_command(
  OUTPUT forward_text_encoder.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_text_encoder_gemmini.mlir
             ${OUTPUT_DIR}/forward_text_encoder_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_text_encoder.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-gemmini
            -o ${OUTPUT_DIR}/forward_text_encoder_gemmini.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_text_encoder.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-gemmini
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/forward_text_encoder_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_text_encoder.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-gemmini
            -o ${OUTPUT_DIR}/forward_text_encoder_gemmini.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_text_encoder.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-gemmini
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${COMPILER_BUILD}/bin/buddy-translate
            --buddy-to-llvmir |
          ${COMPILER_BUILD}/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xgemmini,+D -float-abi=hard -o forward_text_encoder.o
  DEPENDS ${OUTPUT_DIR}/forward_text_encoder.mlir
  COMMENT "Building forward_text_encoder.o "
  VERBATIM
)

add_custom_command(
  OUTPUT subgraph0_text_encoder.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_text_encoder_gemmini.mlir
             ${OUTPUT_DIR}/subgraph0_text_encoder_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-gemmini
            -o ${OUTPUT_DIR}/subgraph0_text_encoder_gemmini.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-gemmini
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/subgraph0_text_encoder_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-gemmini
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${COMPILER_BUILD}/bin/buddy-translate
            --buddy-to-llvmir |
          ${COMPILER_BUILD}/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xgemmini,+D -float-abi=hard -o subgraph0_text_encoder.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_text_encoder.mlir
  COMMENT "Building subgraph0_text_encoder.o "
  VERBATIM
)

add_custom_command(
  OUTPUT forward_unet.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_unet_gemmini.mlir
             ${OUTPUT_DIR}/forward_unet_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_unet.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-gemmini
            -o ${OUTPUT_DIR}/forward_unet_gemmini.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_unet.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-gemmini
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/forward_unet_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_unet.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-gemmini
            -o ${OUTPUT_DIR}/forward_unet_gemmini.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_unet.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-gemmini
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${COMPILER_BUILD}/bin/buddy-translate
            --buddy-to-llvmir |
          ${COMPILER_BUILD}/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xgemmini,+D -float-abi=hard -o forward_unet.o
  DEPENDS ${OUTPUT_DIR}/forward_unet.mlir
  COMMENT "Building forward_unet.o "
  VERBATIM
)

add_custom_command(
  OUTPUT subgraph0_unet.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_unet_gemmini.mlir
             ${OUTPUT_DIR}/subgraph0_unet_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_unet.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-gemmini
            -o ${OUTPUT_DIR}/subgraph0_unet_gemmini.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_unet.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-gemmini
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/subgraph0_unet_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_unet.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-gemmini
            -o ${OUTPUT_DIR}/subgraph0_unet_gemmini.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_unet.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-gemmini
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${COMPILER_BUILD}/bin/buddy-translate
            --buddy-to-llvmir |
          ${COMPILER_BUILD}/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xgemmini,+D -float-abi=hard -o subgraph0_unet.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_unet.mlir
  COMMENT "Building subgraph0_unet.o "
  VERBATIM
)

add_custom_command(
  OUTPUT forward_vae.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_vae_gemmini.mlir
             ${OUTPUT_DIR}/forward_vae_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_vae.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-gemmini
            -o ${OUTPUT_DIR}/forward_vae_gemmini.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_vae.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -lower-gemmini
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/forward_vae_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_vae.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-gemmini
            -o ${OUTPUT_DIR}/forward_vae_gemmini.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_vae.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -lower-gemmini
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${COMPILER_BUILD}/bin/buddy-translate
            --buddy-to-llvmir |
          ${COMPILER_BUILD}/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xgemmini,+D -float-abi=hard -o forward_vae.o
  DEPENDS ${OUTPUT_DIR}/forward_vae.mlir
  COMMENT "Building forward_vae.o "
  VERBATIM
)

add_custom_command(
  OUTPUT subgraph0_vae.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_vae_gemmini.mlir
             ${OUTPUT_DIR}/subgraph0_vae_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_vae.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-gemmini
            -o ${OUTPUT_DIR}/subgraph0_vae_gemmini.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_vae.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-gemmini
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/subgraph0_vae_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_vae.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-gemmini
            -o ${OUTPUT_DIR}/subgraph0_vae_gemmini.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_vae.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-gemmini
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${COMPILER_BUILD}/bin/buddy-translate
            --buddy-to-llvmir |
          ${COMPILER_BUILD}/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xgemmini,+D -float-abi=hard -o subgraph0_vae.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_vae.mlir
  COMMENT "Building subgraph0_vae.o "
  VERBATIM
)

add_library(stable-diffusion-gemmini-textencoder-model STATIC EXCLUDE_FROM_ALL subgraph0_text_encoder.o
  forward_text_encoder.o
)
set_target_properties(stable-diffusion-gemmini-textencoder-model PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
add_library(stable-diffusion-gemmini-unet-model STATIC EXCLUDE_FROM_ALL subgraph0_unet.o forward_unet.o)
set_target_properties(stable-diffusion-gemmini-unet-model PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
add_library(stable-diffusion-gemmini-vae-model STATIC EXCLUDE_FROM_ALL subgraph0_vae.o forward_vae.o)
set_target_properties(stable-diffusion-gemmini-vae-model PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}")

set_target_properties(stable-diffusion-gemmini-textencoder-model PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(stable-diffusion-gemmini-unet-model PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(stable-diffusion-gemmini-vae-model PROPERTIES LINKER_LANGUAGE CXX)

add_executable(buddy-gemmini-stable-diffusion-run EXCLUDE_FROM_ALL ${MODEL_DIR}/stable-diffusion/main.cpp)
set_target_properties(buddy-gemmini-stable-diffusion-run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")

target_include_directories(buddy-gemmini-stable-diffusion-run PRIVATE ${BUDDY_SOURCE}/frontend/Interfaces
                                                                      ${MODEL_DIR}/stable-diffusion/include
)

if(BUDDY_MLIR_USE_MIMALLOC)
  list(APPEND BUDDY_STABLE_DIFFUSION_LIBS mimalloc)
endif()

target_link_libraries(buddy-gemmini-stable-diffusion-run -static stable-diffusion-gemmini-textencoder-model
                                                                 stable-diffusion-gemmini-unet-model
                                                                 stable-diffusion-gemmini-vae-model
                                                                 gemmini-runtime
                                                                 model_image
)

add_custom_target(
  stable-diffusion-model-build
  DEPENDS
    stable-diffusion-gemmini-textencoder-model
    stable-diffusion-gemmini-unet-model
    stable-diffusion-gemmini-vae-model
)
