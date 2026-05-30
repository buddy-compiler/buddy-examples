set(CMAKE_CXX_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

set(MODEL_STABLEDIFFUSION_DIR ${MODEL_DIR}/StableDiffusion)
set(INTERFACE_DIR ${BUDDY_MLIR_DIR}/frontend/Interfaces)
set(INCLUDE_DIR ${MODEL_STABLEDIFFUSION_DIR}/include)

add_custom_command(
  OUTPUT forward_text_encoder.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/forward_text_encoder.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-tile
            -convert-tile-to-buckyball
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-buckyball-to-bank-ssa
            -assign-physical-banks
            -llvm-request-c-wrappers
            -lower-bank-ssa-to-intrinsics
            -lower-buckyball
            > forward_text_encoder_buckyball.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/forward_text_encoder.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-tile
            -convert-tile-to-buckyball
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-buckyball-to-bank-ssa
            -assign-physical-banks
            -llvm-request-c-wrappers
            -lower-bank-ssa-to-intrinsics
            -lower-buckyball
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
        ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
        ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard
          -o forward_text_encoder.o
  DEPENDS ${MODEL_STABLEDIFFUSION_DIR}/forward_text_encoder.mlir
  COMMENT "Building forward_text_encoder.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph0_text_encoder.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_text_encoder.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-tile
            -convert-tile-to-buckyball
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-buckyball-to-bank-ssa
            -assign-physical-banks
            -llvm-request-c-wrappers
            -lower-bank-ssa-to-intrinsics
            -lower-buckyball
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard
            -o subgraph0_text_encoder.o
    DEPENDS ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_text_encoder.mlir
    COMMENT "Building subgraph0_text_encoder.o "
    VERBATIM)

add_custom_command(
  OUTPUT forward_unet.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/forward_unet.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-tile
            -convert-tile-to-buckyball
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-buckyball-to-bank-ssa
            -assign-physical-banks
            -llvm-request-c-wrappers
            -lower-bank-ssa-to-intrinsics
            -lower-buckyball
            > forward_unet_buckyball.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/forward_unet.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-tile
            -convert-tile-to-buckyball
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-buckyball-to-bank-ssa
            -assign-physical-banks
            -llvm-request-c-wrappers
            -lower-bank-ssa-to-intrinsics
            -lower-buckyball
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
        ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
        ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard
          -o forward_unet.o
  DEPENDS ${MODEL_STABLEDIFFUSION_DIR}/forward_unet.mlir
  COMMENT "Building forward_unet.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph0_unet.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_unet.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-tile
            -convert-tile-to-buckyball
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-buckyball-to-bank-ssa
            -assign-physical-banks
            -llvm-request-c-wrappers
            -lower-bank-ssa-to-intrinsics
            -lower-buckyball
            > subgraph0_unet_buckyball.mlir
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_unet.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-tile
            -convert-tile-to-buckyball
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-buckyball-to-bank-ssa
            -assign-physical-banks
            -llvm-request-c-wrappers
            -lower-bank-ssa-to-intrinsics
            -lower-buckyball
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard
            -o subgraph0_unet.o
    DEPENDS ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_unet.mlir
    COMMENT "Building subgraph0_unet.o "
    VERBATIM)

add_custom_command(
  OUTPUT forward_vae.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/forward_vae.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-tile
            -convert-tile-to-buckyball
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-buckyball-to-bank-ssa
            -assign-physical-banks
            -llvm-request-c-wrappers
            -lower-bank-ssa-to-intrinsics
            -lower-buckyball
            > forward_vae_buckyball.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/forward_vae.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-tile
            -convert-tile-to-buckyball
            -lower-buckyball-to-bank-ssa
            -assign-physical-banks
            -llvm-request-c-wrappers
            -lower-bank-ssa-to-intrinsics
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-buckyball
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
        ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
        ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard
          -o forward_vae.o
  DEPENDS ${MODEL_STABLEDIFFUSION_DIR}/forward_vae.mlir
  COMMENT "Building forward_vae.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph0_vae.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_vae.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -eliminate-empty-tensors
              -eliminate-large-zero-constants
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -one-shot-bufferize="bufferize-function-boundaries"
              -llvm-request-c-wrappers
              -convert-linalg-to-tile
            -convert-tile-to-buckyball
              -expand-strided-metadata
              -convert-linalg-to-loops
              -convert-scf-to-cf
              -lower-buckyball-to-bank-ssa
            -assign-physical-banks
            -llvm-request-c-wrappers
            -lower-bank-ssa-to-intrinsics
            -lower-buckyball
              > subgraph0_vae_buckyball.mlir
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_vae.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -eliminate-large-zero-constants
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-linalg-to-tile
            -convert-tile-to-buckyball
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-buckyball-to-bank-ssa
            -assign-physical-banks
            -llvm-request-c-wrappers
            -lower-bank-ssa-to-intrinsics
            -lower-buckyball
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard
            -o subgraph0_vae.o
    DEPENDS ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_vae.mlir
    COMMENT "Building subgraph0_vae.o "
    VERBATIM)


add_library(BuckyballTEXTENCODER STATIC subgraph0_text_encoder.o forward_text_encoder.o)
add_library(BuckyballUNET STATIC subgraph0_unet.o forward_unet.o)
add_library(BuckyballVAE STATIC subgraph0_vae.o forward_vae.o)

set_target_properties(BuckyballTEXTENCODER PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(BuckyballUNET PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(BuckyballVAE PROPERTIES LINKER_LANGUAGE CXX)

add_dependencies(BuckyballTEXTENCODER stablediffusion-model-build)
add_dependencies(BuckyballUNET stablediffusion-model-build)
add_dependencies(BuckyballVAE stablediffusion-model-build)

add_executable(buddy-buckyball-stablediffusion-run ${MODEL_STABLEDIFFUSION_DIR}/buddy-stable-diffusion-main.cpp)

target_include_directories(buddy-buckyball-stablediffusion-run PRIVATE ${INTERFACE_DIR} ${INCLUDE_DIR})


if(BUDDY_MLIR_USE_MIMALLOC)
  list(APPEND BUDDY_STABLE_DIFFUSION_LIBS mimalloc)
endif()

set(BUDDY_STABLE_DIFFUSION_LIBS
  BuckyballTEXTENCODER
  BuckyballUNET
  BuckyballVAE
  ModelTestCRunnerUtils
  ModelTestDIP_riscv
)

target_link_libraries(buddy-buckyball-stablediffusion-run -static ${BUDDY_STABLE_DIFFUSION_LIBS})

set(STABLEDIFFUSION_OUTPUT_DIR ${OUTPUT_BIN_DIR}/src/ModelTest/e2e/models/archs/buckyball/StableDiffusion)

add_custom_command(TARGET buddy-buckyball-stablediffusion-run POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${STABLEDIFFUSION_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:buddy-buckyball-stablediffusion-run>
          ${STABLEDIFFUSION_OUTPUT_DIR}/buddy-buckyball-stablediffusion-run
  COMMENT "Copying Buckyball StableDiffusion executable to output directory"
  VERBATIM
)

add_custom_target(sync-buckyball-stablediffusion-deps
  COMMAND ${CMAKE_COMMAND} -E make_directory ${STABLEDIFFUSION_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_STABLEDIFFUSION_DIR}/arg0_text_encoder.data
          ${STABLEDIFFUSION_OUTPUT_DIR}/arg0_text_encoder.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_STABLEDIFFUSION_DIR}/arg1_text_encoder.data
          ${STABLEDIFFUSION_OUTPUT_DIR}/arg1_text_encoder.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_STABLEDIFFUSION_DIR}/arg0_unet.data
          ${STABLEDIFFUSION_OUTPUT_DIR}/arg0_unet.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_STABLEDIFFUSION_DIR}/arg0_vae.data
          ${STABLEDIFFUSION_OUTPUT_DIR}/arg0_vae.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_STABLEDIFFUSION_DIR}/vocab.txt
          ${STABLEDIFFUSION_OUTPUT_DIR}/vocab.txt
  DEPENDS
          ${MODEL_STABLEDIFFUSION_DIR}/arg0_text_encoder.data
          ${MODEL_STABLEDIFFUSION_DIR}/arg1_text_encoder.data
          ${MODEL_STABLEDIFFUSION_DIR}/arg0_unet.data
          ${MODEL_STABLEDIFFUSION_DIR}/arg0_vae.data
          ${MODEL_STABLEDIFFUSION_DIR}/vocab.txt
  COMMENT "Syncing StableDiffusion runtime deps to output directory"
  VERBATIM
)

add_dependencies(buddy-buckyball-stablediffusion-run sync-buckyball-stablediffusion-deps)
