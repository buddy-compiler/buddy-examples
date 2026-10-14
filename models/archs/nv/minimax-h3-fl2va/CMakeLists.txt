set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

add_custom_command(
  OUTPUT ${OUTPUT_DIR}/arg0_text_encoder.data
         ${OUTPUT_DIR}/arg0_transformer.data
         ${OUTPUT_DIR}/arg0_visual_vae.data
         ${OUTPUT_DIR}/arg0_audio_vae.data
         ${OUTPUT_DIR}/forward_text_encoder.mlir
         ${OUTPUT_DIR}/subgraph0_text_encoder.mlir
         ${OUTPUT_DIR}/forward_transformer.mlir
         ${OUTPUT_DIR}/subgraph0_transformer.mlir
         ${OUTPUT_DIR}/forward_visual_vae.mlir
         ${OUTPUT_DIR}/subgraph0_visual_vae.mlir
         ${OUTPUT_DIR}/forward_visual_vae_encode.mlir
         ${OUTPUT_DIR}/subgraph0_visual_vae_encode.mlir
         ${OUTPUT_DIR}/forward_audio_vae.mlir
         ${OUTPUT_DIR}/subgraph0_audio_vae.mlir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
  COMMAND ${PYTHON} buddy-pytorch-importer.py
            --repo-root ${REPO_ROOT}
            --output-dir ${OUTPUT_DIR}
            --compiler-build ${COMPILER_BUILD}
            --duration-s 4
            --fps 24
            --height 768
            --width 1344
            --latent-c 24
            --audio-latent-c 32
            --text-dim 5120
            --text-len 512
            --sample-rate 32000
            --denoise-steps 30
            --video-shift 12.0
            --frames 96
            --latent-t 24
            --latent-h 48
            --latent-w 84
            --audio-latent-t 160
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing minimax-h3-fl2va for nv"
  VERBATIM
)

file(CONFIGURE OUTPUT "${OUTPUT_DIR}/model_parameters.h" CONTENT
  [=[#pragma once
namespace model_parameters {
inline constexpr int DURATION_S = 4;
inline constexpr int FPS = 24;
inline constexpr int HEIGHT = 768;
inline constexpr int WIDTH = 1344;
inline constexpr int LATENT_C = 24;
inline constexpr int AUDIO_LATENT_C = 32;
inline constexpr int TEXT_DIM = 5120;
inline constexpr int TEXT_LEN = 512;
inline constexpr int SAMPLE_RATE = 32000;
inline constexpr int DENOISE_STEPS = 30;
inline constexpr float VIDEO_SHIFT = static_cast<float>(12.0);
inline constexpr int FRAMES = 96;
inline constexpr int LATENT_T = 24;
inline constexpr int LATENT_H = 48;
inline constexpr int LATENT_W = 84;
inline constexpr int AUDIO_LATENT_T = 160;
}
]=]
)
add_custom_command(
  OUTPUT forward_text_encoder_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_text_encoder_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_text_encoder.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -buffer-hoisting
            -canonicalize
            -cse
            -expand-strided-metadata
            -convert-memcpy-to-gpu
            -gpu-async-region |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers
            --gpu-to-llvm
            -o ${OUTPUT_DIR}/forward_text_encoder_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_text_encoder.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -buffer-hoisting
            -canonicalize
            -cse
            -expand-strided-metadata
            -convert-memcpy-to-gpu
            -gpu-async-region |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers
            --gpu-to-llvm |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-translate -mlir-to-llvmir |
          ${BUDDY_SOURCE}/llvm/build/bin/llvm-as |
          ${BUDDY_SOURCE}/llvm/build/bin/llc -filetype=obj -relocation-model=pic -O2 -o ${CMAKE_CURRENT_BINARY_DIR}/forward_text_encoder_gpu.o
  DEPENDS ${OUTPUT_DIR}/forward_text_encoder.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
  COMMENT "Building NV forward_text_encoder_gpu.o"
  VERBATIM
)

add_custom_command(
  OUTPUT subgraph0_text_encoder_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_gpu.mlir
             ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_nvvm.mlir
             ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-strided-memref-copy-to-linalg
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -o ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-strided-memref-copy-to-linalg
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse |
          ${COMPILER_BUILD}/bin/buddy-opt
            -convert-memcpy-to-gpu=process-args=0
            -gpu-async-region
            -canonicalize |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin"
            -o ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_nvvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-strided-memref-copy-to-linalg
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse |
          ${COMPILER_BUILD}/bin/buddy-opt
            -convert-memcpy-to-gpu=process-args=0
            -gpu-async-region
            -canonicalize |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin"
            -o ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-strided-memref-copy-to-linalg
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse |
          ${COMPILER_BUILD}/bin/buddy-opt
            -convert-memcpy-to-gpu=process-args=0
            -gpu-async-region
            -canonicalize |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin" |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-translate -mlir-to-llvmir |
          ${BUDDY_SOURCE}/llvm/build/bin/llvm-as |
          ${BUDDY_SOURCE}/llvm/build/bin/llc -filetype=obj -relocation-model=pic -O2 -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_text_encoder_gpu.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_text_encoder.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
  COMMENT "Building NV subgraph0_text_encoder_gpu.o"
  VERBATIM
)

add_custom_command(
  OUTPUT forward_transformer_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_transformer_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_transformer.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -buffer-hoisting
            -canonicalize
            -cse
            -expand-strided-metadata
            -convert-memcpy-to-gpu
            -gpu-async-region |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers
            --gpu-to-llvm
            -o ${OUTPUT_DIR}/forward_transformer_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_transformer.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -buffer-hoisting
            -canonicalize
            -cse
            -expand-strided-metadata
            -convert-memcpy-to-gpu
            -gpu-async-region |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers
            --gpu-to-llvm |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-translate -mlir-to-llvmir |
          ${BUDDY_SOURCE}/llvm/build/bin/llvm-as |
          ${BUDDY_SOURCE}/llvm/build/bin/llc -filetype=obj -relocation-model=pic -O2 -o ${CMAKE_CURRENT_BINARY_DIR}/forward_transformer_gpu.o
  DEPENDS ${OUTPUT_DIR}/forward_transformer.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
  COMMENT "Building NV forward_transformer_gpu.o"
  VERBATIM
)

add_custom_command(
  OUTPUT subgraph0_transformer_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_transformer_gpu_gpu.mlir
             ${OUTPUT_DIR}/subgraph0_transformer_gpu_nvvm.mlir
             ${OUTPUT_DIR}/subgraph0_transformer_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_transformer.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-strided-memref-copy-to-linalg
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -o ${OUTPUT_DIR}/subgraph0_transformer_gpu_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_transformer.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-strided-memref-copy-to-linalg
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse |
          ${COMPILER_BUILD}/bin/buddy-opt
            -convert-memcpy-to-gpu=process-args=0
            -gpu-async-region
            -canonicalize |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin"
            -o ${OUTPUT_DIR}/subgraph0_transformer_gpu_nvvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_transformer.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-strided-memref-copy-to-linalg
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse |
          ${COMPILER_BUILD}/bin/buddy-opt
            -convert-memcpy-to-gpu=process-args=0
            -gpu-async-region
            -canonicalize |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin"
            -o ${OUTPUT_DIR}/subgraph0_transformer_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_transformer.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-strided-memref-copy-to-linalg
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse |
          ${COMPILER_BUILD}/bin/buddy-opt
            -convert-memcpy-to-gpu=process-args=0
            -gpu-async-region
            -canonicalize |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin" |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-translate -mlir-to-llvmir |
          ${BUDDY_SOURCE}/llvm/build/bin/llvm-as |
          ${BUDDY_SOURCE}/llvm/build/bin/llc -filetype=obj -relocation-model=pic -O2 -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_transformer_gpu.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_transformer.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
  COMMENT "Building NV subgraph0_transformer_gpu.o"
  VERBATIM
)

add_custom_command(
  OUTPUT forward_visual_vae_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_visual_vae_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_visual_vae.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -buffer-hoisting
            -canonicalize
            -cse
            -expand-strided-metadata
            -convert-memcpy-to-gpu
            -gpu-async-region |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers
            --gpu-to-llvm
            -o ${OUTPUT_DIR}/forward_visual_vae_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_visual_vae.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -buffer-hoisting
            -canonicalize
            -cse
            -expand-strided-metadata
            -convert-memcpy-to-gpu
            -gpu-async-region |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers
            --gpu-to-llvm |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-translate -mlir-to-llvmir |
          ${BUDDY_SOURCE}/llvm/build/bin/llvm-as |
          ${BUDDY_SOURCE}/llvm/build/bin/llc -filetype=obj -relocation-model=pic -O2 -o ${CMAKE_CURRENT_BINARY_DIR}/forward_visual_vae_gpu.o
  DEPENDS ${OUTPUT_DIR}/forward_visual_vae.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
  COMMENT "Building NV forward_visual_vae_gpu.o"
  VERBATIM
)

add_custom_command(
  OUTPUT subgraph0_visual_vae_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_visual_vae_gpu_gpu.mlir
             ${OUTPUT_DIR}/subgraph0_visual_vae_gpu_nvvm.mlir
             ${OUTPUT_DIR}/subgraph0_visual_vae_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_visual_vae.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-strided-memref-copy-to-linalg
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -o ${OUTPUT_DIR}/subgraph0_visual_vae_gpu_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_visual_vae.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-strided-memref-copy-to-linalg
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse |
          ${COMPILER_BUILD}/bin/buddy-opt
            -convert-memcpy-to-gpu=process-args=0
            -gpu-async-region
            -canonicalize |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin"
            -o ${OUTPUT_DIR}/subgraph0_visual_vae_gpu_nvvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_visual_vae.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-strided-memref-copy-to-linalg
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse |
          ${COMPILER_BUILD}/bin/buddy-opt
            -convert-memcpy-to-gpu=process-args=0
            -gpu-async-region
            -canonicalize |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin"
            -o ${OUTPUT_DIR}/subgraph0_visual_vae_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_visual_vae.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-strided-memref-copy-to-linalg
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse |
          ${COMPILER_BUILD}/bin/buddy-opt
            -convert-memcpy-to-gpu=process-args=0
            -gpu-async-region
            -canonicalize |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin" |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-translate -mlir-to-llvmir |
          ${BUDDY_SOURCE}/llvm/build/bin/llvm-as |
          ${BUDDY_SOURCE}/llvm/build/bin/llc -filetype=obj -relocation-model=pic -O2 -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_visual_vae_gpu.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_visual_vae.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
  COMMENT "Building NV subgraph0_visual_vae_gpu.o"
  VERBATIM
)

add_custom_command(
  OUTPUT forward_visual_vae_encode_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_visual_vae_encode_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_visual_vae_encode.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -buffer-hoisting
            -canonicalize
            -cse
            -expand-strided-metadata
            -convert-memcpy-to-gpu
            -gpu-async-region |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers
            --gpu-to-llvm
            -o ${OUTPUT_DIR}/forward_visual_vae_encode_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_visual_vae_encode.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -buffer-hoisting
            -canonicalize
            -cse
            -expand-strided-metadata
            -convert-memcpy-to-gpu
            -gpu-async-region |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers
            --gpu-to-llvm |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-translate -mlir-to-llvmir |
          ${BUDDY_SOURCE}/llvm/build/bin/llvm-as |
          ${BUDDY_SOURCE}/llvm/build/bin/llc -filetype=obj -relocation-model=pic -O2 -o ${CMAKE_CURRENT_BINARY_DIR}/forward_visual_vae_encode_gpu.o
  DEPENDS ${OUTPUT_DIR}/forward_visual_vae_encode.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
  COMMENT "Building NV forward_visual_vae_encode_gpu.o"
  VERBATIM
)

add_custom_command(
  OUTPUT subgraph0_visual_vae_encode_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_visual_vae_encode_gpu_gpu.mlir
             ${OUTPUT_DIR}/subgraph0_visual_vae_encode_gpu_nvvm.mlir
             ${OUTPUT_DIR}/subgraph0_visual_vae_encode_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_visual_vae_encode.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-strided-memref-copy-to-linalg
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -o ${OUTPUT_DIR}/subgraph0_visual_vae_encode_gpu_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_visual_vae_encode.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-strided-memref-copy-to-linalg
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse |
          ${COMPILER_BUILD}/bin/buddy-opt
            -convert-memcpy-to-gpu=process-args=0
            -gpu-async-region
            -canonicalize |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin"
            -o ${OUTPUT_DIR}/subgraph0_visual_vae_encode_gpu_nvvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_visual_vae_encode.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-strided-memref-copy-to-linalg
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse |
          ${COMPILER_BUILD}/bin/buddy-opt
            -convert-memcpy-to-gpu=process-args=0
            -gpu-async-region
            -canonicalize |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin"
            -o ${OUTPUT_DIR}/subgraph0_visual_vae_encode_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_visual_vae_encode.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-strided-memref-copy-to-linalg
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse |
          ${COMPILER_BUILD}/bin/buddy-opt
            -convert-memcpy-to-gpu=process-args=0
            -gpu-async-region
            -canonicalize |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin" |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-translate -mlir-to-llvmir |
          ${BUDDY_SOURCE}/llvm/build/bin/llvm-as |
          ${BUDDY_SOURCE}/llvm/build/bin/llc -filetype=obj -relocation-model=pic -O2 -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_visual_vae_encode_gpu.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_visual_vae_encode.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
  COMMENT "Building NV subgraph0_visual_vae_encode_gpu.o"
  VERBATIM
)

add_custom_command(
  OUTPUT forward_audio_vae_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_audio_vae_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_audio_vae.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -buffer-hoisting
            -canonicalize
            -cse
            -expand-strided-metadata
            -convert-memcpy-to-gpu
            -gpu-async-region |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers
            --gpu-to-llvm
            -o ${OUTPUT_DIR}/forward_audio_vae_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_audio_vae.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -buffer-hoisting
            -canonicalize
            -cse
            -expand-strided-metadata
            -convert-memcpy-to-gpu
            -gpu-async-region |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers
            --gpu-to-llvm |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-translate -mlir-to-llvmir |
          ${BUDDY_SOURCE}/llvm/build/bin/llvm-as |
          ${BUDDY_SOURCE}/llvm/build/bin/llc -filetype=obj -relocation-model=pic -O2 -o ${CMAKE_CURRENT_BINARY_DIR}/forward_audio_vae_gpu.o
  DEPENDS ${OUTPUT_DIR}/forward_audio_vae.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
  COMMENT "Building NV forward_audio_vae_gpu.o"
  VERBATIM
)

add_custom_command(
  OUTPUT subgraph0_audio_vae_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_audio_vae_gpu_gpu.mlir
             ${OUTPUT_DIR}/subgraph0_audio_vae_gpu_nvvm.mlir
             ${OUTPUT_DIR}/subgraph0_audio_vae_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_audio_vae.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-strided-memref-copy-to-linalg
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -o ${OUTPUT_DIR}/subgraph0_audio_vae_gpu_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_audio_vae.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-strided-memref-copy-to-linalg
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse |
          ${COMPILER_BUILD}/bin/buddy-opt
            -convert-memcpy-to-gpu=process-args=0
            -gpu-async-region
            -canonicalize |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin"
            -o ${OUTPUT_DIR}/subgraph0_audio_vae_gpu_nvvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_audio_vae.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-strided-memref-copy-to-linalg
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse |
          ${COMPILER_BUILD}/bin/buddy-opt
            -convert-memcpy-to-gpu=process-args=0
            -gpu-async-region
            -canonicalize |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin"
            -o ${OUTPUT_DIR}/subgraph0_audio_vae_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_audio_vae.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-strided-memref-copy-to-linalg
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse |
          ${COMPILER_BUILD}/bin/buddy-opt
            -convert-memcpy-to-gpu=process-args=0
            -gpu-async-region
            -canonicalize |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin" |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-translate -mlir-to-llvmir |
          ${BUDDY_SOURCE}/llvm/build/bin/llvm-as |
          ${BUDDY_SOURCE}/llvm/build/bin/llc -filetype=obj -relocation-model=pic -O2 -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_audio_vae_gpu.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_audio_vae.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
  COMMENT "Building NV subgraph0_audio_vae_gpu.o"
  VERBATIM
)

add_library(minimax-h3-fl2va-nv-text-encoder-model STATIC EXCLUDE_FROM_ALL subgraph0_text_encoder_gpu.o
  forward_text_encoder_gpu.o
)
set_target_properties(minimax-h3-fl2va-nv-text-encoder-model PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
set_target_properties(minimax-h3-fl2va-nv-text-encoder-model PROPERTIES LINKER_LANGUAGE CXX)
add_library(minimax-h3-fl2va-nv-transformer-model STATIC EXCLUDE_FROM_ALL subgraph0_transformer_gpu.o
  forward_transformer_gpu.o
)
set_target_properties(minimax-h3-fl2va-nv-transformer-model PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
set_target_properties(minimax-h3-fl2va-nv-transformer-model PROPERTIES LINKER_LANGUAGE CXX)
add_library(minimax-h3-fl2va-nv-visual-vae-model STATIC EXCLUDE_FROM_ALL subgraph0_visual_vae_gpu.o
  forward_visual_vae_gpu.o
)
set_target_properties(minimax-h3-fl2va-nv-visual-vae-model PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
set_target_properties(minimax-h3-fl2va-nv-visual-vae-model PROPERTIES LINKER_LANGUAGE CXX)
add_library(minimax-h3-fl2va-nv-visual-vae-encode-model STATIC EXCLUDE_FROM_ALL subgraph0_visual_vae_encode_gpu.o
  forward_visual_vae_encode_gpu.o
)
set_target_properties(minimax-h3-fl2va-nv-visual-vae-encode-model PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
set_target_properties(minimax-h3-fl2va-nv-visual-vae-encode-model PROPERTIES LINKER_LANGUAGE CXX)
add_library(minimax-h3-fl2va-nv-audio-vae-model STATIC EXCLUDE_FROM_ALL subgraph0_audio_vae_gpu.o
  forward_audio_vae_gpu.o
)
set_target_properties(minimax-h3-fl2va-nv-audio-vae-model PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
set_target_properties(minimax-h3-fl2va-nv-audio-vae-model PROPERTIES LINKER_LANGUAGE CXX)

add_executable(buddy-nv-minimax-h3-fl2va-run EXCLUDE_FROM_ALL ${MODEL_DIR}/minimax-h3-fl2va/main.cpp)
set_target_properties(buddy-nv-minimax-h3-fl2va-run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
add_dependencies(
  buddy-nv-minimax-h3-fl2va-run
  minimax-h3-fl2va-nv-text-encoder-model
  minimax-h3-fl2va-nv-transformer-model
  minimax-h3-fl2va-nv-visual-vae-model
  minimax-h3-fl2va-nv-visual-vae-encode-model
  minimax-h3-fl2va-nv-audio-vae-model
)
target_compile_definitions(
  buddy-nv-minimax-h3-fl2va-run
  PRIVATE MINIMAX_H3_FL2VA_EXAMPLE_PATH="${MODEL_DIR}/minimax-h3-fl2va/"
)
target_include_directories(buddy-nv-minimax-h3-fl2va-run PRIVATE ${BUDDY_SOURCE}/frontend/Interfaces
                                                                 "${OUTPUT_DIR}"
)
target_link_directories(buddy-nv-minimax-h3-fl2va-run PRIVATE ${BUDDY_SOURCE}/llvm/build/lib ${BUDDY_SOURCE}/build/lib)
target_link_libraries(buddy-nv-minimax-h3-fl2va-run minimax-h3-fl2va-nv-text-encoder-model minimax-h3-fl2va-nv-transformer-model
                                                                                           minimax-h3-fl2va-nv-visual-vae-model
                                                                                           minimax-h3-fl2va-nv-visual-vae-encode-model
                                                                                           minimax-h3-fl2va-nv-audio-vae-model
                                                                                           nv-runtime
                                                                                           mlir_c_runner_utils
                                                                                           mlir_runner_utils
                                                                                           mlir_async_runtime
                                                                                           mlir_cuda_runtime
                                                                                           dl
                                                                                           pthread
)

add_custom_target(
  minimax-h3-fl2va-model-build
  DEPENDS
    minimax-h3-fl2va-nv-text-encoder-model
    minimax-h3-fl2va-nv-transformer-model
    minimax-h3-fl2va-nv-visual-vae-model
    minimax-h3-fl2va-nv-visual-vae-encode-model
    minimax-h3-fl2va-nv-audio-vae-model
)
