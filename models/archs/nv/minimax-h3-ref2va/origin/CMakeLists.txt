set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

add_custom_command(
  OUTPUT ${OUTPUT_DIR}/arg0_text_encoder.data
         ${OUTPUT_DIR}/arg0_transformer.data
         ${OUTPUT_DIR}/arg0_visual_vae.data
         ${OUTPUT_DIR}/arg0_visual_vae_encode.data
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
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing minimax-h3-ref2va for nv"
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

# ===----------------------------------------------------------------------===
# Compile the host entry point. Each command must succeed before the next runs.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward_text_encoder_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_text_encoder_trace.mlir
             ${OUTPUT_DIR}/forward_text_encoder_linalg.mlir
             ${OUTPUT_DIR}/forward_text_encoder_gpu.mlir
             ${OUTPUT_DIR}/forward_text_encoder_gpu_llvm.mlir
             ${OUTPUT_DIR}/forward_text_encoder_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_text_encoder.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/forward_text_encoder_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_text_encoder_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/forward_text_encoder_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_text_encoder_linalg.mlir
            -pass-pipeline "builtin.module(func.func(buffer-hoisting),canonicalize,cse,expand-strided-metadata,func.func(convert-memcpy-to-gpu,gpu-async-region))"
            -o ${OUTPUT_DIR}/forward_text_encoder_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_text_encoder_gpu.mlir
            -llvm-request-c-wrappers
            -gpu-to-llvm
            -o ${OUTPUT_DIR}/forward_text_encoder_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/forward_text_encoder_gpu_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/forward_text_encoder_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward_text_encoder_gpu.ll -filetype=obj -relocation-model=pic -O2 -o forward_text_encoder_gpu.o
  DEPENDS ${OUTPUT_DIR}/forward_text_encoder.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling minimax-h3-ref2va host entry point"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower model computation to GPU kernels and embed the NVIDIA binary.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0_text_encoder_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_text_encoder_trace.mlir
             ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_gpu.mlir
             ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_async.mlir
             ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_llvm.mlir
             ${OUTPUT_DIR}/subgraph0_text_encoder_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/subgraph0_text_encoder_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_linalg.mlir
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse
            -o ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_gpu.mlir
            -pass-pipeline "builtin.module(func.func(convert-memcpy-to-gpu{process-args=0},gpu-async-region),canonicalize)"
            -o ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_async.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_async.mlir
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=${NV_CUBIN_VERSION} cubin-features=+ptx71 cubin-format=fatbin"
            -o ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/subgraph0_text_encoder_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0_text_encoder_gpu.ll -filetype=obj -relocation-model=pic -O2 -o subgraph0_text_encoder_gpu.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_text_encoder.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling minimax-h3-ref2va NVIDIA kernels"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Compile the host entry point. Each command must succeed before the next runs.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward_transformer_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_transformer_trace.mlir
             ${OUTPUT_DIR}/forward_transformer_linalg.mlir
             ${OUTPUT_DIR}/forward_transformer_gpu.mlir
             ${OUTPUT_DIR}/forward_transformer_gpu_llvm.mlir
             ${OUTPUT_DIR}/forward_transformer_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_transformer.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/forward_transformer_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_transformer_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/forward_transformer_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_transformer_linalg.mlir
            -pass-pipeline "builtin.module(func.func(buffer-hoisting),canonicalize,cse,expand-strided-metadata,func.func(convert-memcpy-to-gpu,gpu-async-region))"
            -o ${OUTPUT_DIR}/forward_transformer_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_transformer_gpu.mlir
            -llvm-request-c-wrappers
            -gpu-to-llvm
            -o ${OUTPUT_DIR}/forward_transformer_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/forward_transformer_gpu_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/forward_transformer_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward_transformer_gpu.ll -filetype=obj -relocation-model=pic -O2 -o forward_transformer_gpu.o
  DEPENDS ${OUTPUT_DIR}/forward_transformer.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling minimax-h3-ref2va host entry point"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower model computation to GPU kernels and embed the NVIDIA binary.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0_transformer_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_transformer_trace.mlir
             ${OUTPUT_DIR}/subgraph0_transformer_gpu_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_transformer_gpu_gpu.mlir
             ${OUTPUT_DIR}/subgraph0_transformer_gpu_async.mlir
             ${OUTPUT_DIR}/subgraph0_transformer_gpu_llvm.mlir
             ${OUTPUT_DIR}/subgraph0_transformer_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_transformer.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/subgraph0_transformer_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_transformer_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_transformer_gpu_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_transformer_gpu_linalg.mlir
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse
            -o ${OUTPUT_DIR}/subgraph0_transformer_gpu_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_transformer_gpu_gpu.mlir
            -pass-pipeline "builtin.module(func.func(convert-memcpy-to-gpu{process-args=0},gpu-async-region),canonicalize)"
            -o ${OUTPUT_DIR}/subgraph0_transformer_gpu_async.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_transformer_gpu_async.mlir
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=${NV_CUBIN_VERSION} cubin-features=+ptx71 cubin-format=fatbin"
            -o ${OUTPUT_DIR}/subgraph0_transformer_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_transformer_gpu_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/subgraph0_transformer_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0_transformer_gpu.ll -filetype=obj -relocation-model=pic -O2 -o subgraph0_transformer_gpu.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_transformer.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling minimax-h3-ref2va NVIDIA kernels"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Compile the host entry point. Each command must succeed before the next runs.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward_visual_vae_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_visual_vae_trace.mlir
             ${OUTPUT_DIR}/forward_visual_vae_linalg.mlir
             ${OUTPUT_DIR}/forward_visual_vae_gpu.mlir
             ${OUTPUT_DIR}/forward_visual_vae_gpu_llvm.mlir
             ${OUTPUT_DIR}/forward_visual_vae_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_visual_vae.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/forward_visual_vae_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_visual_vae_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/forward_visual_vae_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_visual_vae_linalg.mlir
            -pass-pipeline "builtin.module(func.func(buffer-hoisting),canonicalize,cse,expand-strided-metadata,func.func(convert-memcpy-to-gpu,gpu-async-region))"
            -o ${OUTPUT_DIR}/forward_visual_vae_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_visual_vae_gpu.mlir
            -llvm-request-c-wrappers
            -gpu-to-llvm
            -o ${OUTPUT_DIR}/forward_visual_vae_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/forward_visual_vae_gpu_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/forward_visual_vae_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward_visual_vae_gpu.ll -filetype=obj -relocation-model=pic -O2 -o forward_visual_vae_gpu.o
  DEPENDS ${OUTPUT_DIR}/forward_visual_vae.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling minimax-h3-ref2va host entry point"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower model computation to GPU kernels and embed the NVIDIA binary.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0_visual_vae_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_visual_vae_trace.mlir
             ${OUTPUT_DIR}/subgraph0_visual_vae_gpu_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_visual_vae_gpu_gpu.mlir
             ${OUTPUT_DIR}/subgraph0_visual_vae_gpu_async.mlir
             ${OUTPUT_DIR}/subgraph0_visual_vae_gpu_llvm.mlir
             ${OUTPUT_DIR}/subgraph0_visual_vae_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_visual_vae.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/subgraph0_visual_vae_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_visual_vae_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_visual_vae_gpu_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_visual_vae_gpu_linalg.mlir
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse
            -o ${OUTPUT_DIR}/subgraph0_visual_vae_gpu_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_visual_vae_gpu_gpu.mlir
            -pass-pipeline "builtin.module(func.func(convert-memcpy-to-gpu{process-args=0},gpu-async-region),canonicalize)"
            -o ${OUTPUT_DIR}/subgraph0_visual_vae_gpu_async.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_visual_vae_gpu_async.mlir
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=${NV_CUBIN_VERSION} cubin-features=+ptx71 cubin-format=fatbin"
            -o ${OUTPUT_DIR}/subgraph0_visual_vae_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_visual_vae_gpu_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/subgraph0_visual_vae_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0_visual_vae_gpu.ll -filetype=obj -relocation-model=pic -O2 -o subgraph0_visual_vae_gpu.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_visual_vae.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling minimax-h3-ref2va NVIDIA kernels"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Compile the host entry point. Each command must succeed before the next runs.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward_visual_vae_encode_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_visual_vae_encode_trace.mlir
             ${OUTPUT_DIR}/forward_visual_vae_encode_linalg.mlir
             ${OUTPUT_DIR}/forward_visual_vae_encode_gpu.mlir
             ${OUTPUT_DIR}/forward_visual_vae_encode_gpu_llvm.mlir
             ${OUTPUT_DIR}/forward_visual_vae_encode_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_visual_vae_encode.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/forward_visual_vae_encode_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_visual_vae_encode_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/forward_visual_vae_encode_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_visual_vae_encode_linalg.mlir
            -pass-pipeline "builtin.module(func.func(buffer-hoisting),canonicalize,cse,expand-strided-metadata,func.func(convert-memcpy-to-gpu,gpu-async-region))"
            -o ${OUTPUT_DIR}/forward_visual_vae_encode_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_visual_vae_encode_gpu.mlir
            -llvm-request-c-wrappers
            -gpu-to-llvm
            -o ${OUTPUT_DIR}/forward_visual_vae_encode_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/forward_visual_vae_encode_gpu_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/forward_visual_vae_encode_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward_visual_vae_encode_gpu.ll -filetype=obj -relocation-model=pic -O2 -o forward_visual_vae_encode_gpu.o
  DEPENDS ${OUTPUT_DIR}/forward_visual_vae_encode.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling minimax-h3-ref2va host entry point"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower model computation to GPU kernels and embed the NVIDIA binary.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0_visual_vae_encode_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_visual_vae_encode_trace.mlir
             ${OUTPUT_DIR}/subgraph0_visual_vae_encode_gpu_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_visual_vae_encode_gpu_gpu.mlir
             ${OUTPUT_DIR}/subgraph0_visual_vae_encode_gpu_async.mlir
             ${OUTPUT_DIR}/subgraph0_visual_vae_encode_gpu_llvm.mlir
             ${OUTPUT_DIR}/subgraph0_visual_vae_encode_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_visual_vae_encode.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/subgraph0_visual_vae_encode_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_visual_vae_encode_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_visual_vae_encode_gpu_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_visual_vae_encode_gpu_linalg.mlir
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse
            -o ${OUTPUT_DIR}/subgraph0_visual_vae_encode_gpu_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_visual_vae_encode_gpu_gpu.mlir
            -pass-pipeline "builtin.module(func.func(convert-memcpy-to-gpu{process-args=0},gpu-async-region),canonicalize)"
            -o ${OUTPUT_DIR}/subgraph0_visual_vae_encode_gpu_async.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_visual_vae_encode_gpu_async.mlir
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=${NV_CUBIN_VERSION} cubin-features=+ptx71 cubin-format=fatbin"
            -o ${OUTPUT_DIR}/subgraph0_visual_vae_encode_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_visual_vae_encode_gpu_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/subgraph0_visual_vae_encode_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0_visual_vae_encode_gpu.ll -filetype=obj -relocation-model=pic -O2 -o subgraph0_visual_vae_encode_gpu.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_visual_vae_encode.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling minimax-h3-ref2va NVIDIA kernels"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Compile the host entry point. Each command must succeed before the next runs.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward_audio_vae_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_audio_vae_trace.mlir
             ${OUTPUT_DIR}/forward_audio_vae_linalg.mlir
             ${OUTPUT_DIR}/forward_audio_vae_gpu.mlir
             ${OUTPUT_DIR}/forward_audio_vae_gpu_llvm.mlir
             ${OUTPUT_DIR}/forward_audio_vae_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_audio_vae.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/forward_audio_vae_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_audio_vae_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/forward_audio_vae_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_audio_vae_linalg.mlir
            -pass-pipeline "builtin.module(func.func(buffer-hoisting),canonicalize,cse,expand-strided-metadata,func.func(convert-memcpy-to-gpu,gpu-async-region))"
            -o ${OUTPUT_DIR}/forward_audio_vae_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_audio_vae_gpu.mlir
            -llvm-request-c-wrappers
            -gpu-to-llvm
            -o ${OUTPUT_DIR}/forward_audio_vae_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/forward_audio_vae_gpu_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/forward_audio_vae_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward_audio_vae_gpu.ll -filetype=obj -relocation-model=pic -O2 -o forward_audio_vae_gpu.o
  DEPENDS ${OUTPUT_DIR}/forward_audio_vae.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling minimax-h3-ref2va host entry point"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower model computation to GPU kernels and embed the NVIDIA binary.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0_audio_vae_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_audio_vae_trace.mlir
             ${OUTPUT_DIR}/subgraph0_audio_vae_gpu_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_audio_vae_gpu_gpu.mlir
             ${OUTPUT_DIR}/subgraph0_audio_vae_gpu_async.mlir
             ${OUTPUT_DIR}/subgraph0_audio_vae_gpu_llvm.mlir
             ${OUTPUT_DIR}/subgraph0_audio_vae_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_audio_vae.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/subgraph0_audio_vae_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_audio_vae_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_audio_vae_gpu_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_audio_vae_gpu_linalg.mlir
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse
            -o ${OUTPUT_DIR}/subgraph0_audio_vae_gpu_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_audio_vae_gpu_gpu.mlir
            -pass-pipeline "builtin.module(func.func(convert-memcpy-to-gpu{process-args=0},gpu-async-region),canonicalize)"
            -o ${OUTPUT_DIR}/subgraph0_audio_vae_gpu_async.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_audio_vae_gpu_async.mlir
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=${NV_CUBIN_VERSION} cubin-features=+ptx71 cubin-format=fatbin"
            -o ${OUTPUT_DIR}/subgraph0_audio_vae_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_audio_vae_gpu_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/subgraph0_audio_vae_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0_audio_vae_gpu.ll -filetype=obj -relocation-model=pic -O2 -o subgraph0_audio_vae_gpu.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_audio_vae.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling minimax-h3-ref2va NVIDIA kernels"
  VERBATIM
)

add_library(minimax-h3-ref2va-nv-origin-model SHARED EXCLUDE_FROM_ALL forward_text_encoder_gpu.o
  subgraph0_text_encoder_gpu.o forward_transformer_gpu.o subgraph0_transformer_gpu.o
  forward_visual_vae_gpu.o subgraph0_visual_vae_gpu.o forward_visual_vae_encode_gpu.o
  subgraph0_visual_vae_encode_gpu.o forward_audio_vae_gpu.o subgraph0_audio_vae_gpu.o
)
set_target_properties(minimax-h3-ref2va-nv-origin-model
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             LINKER_LANGUAGE CXX
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_link_directories(minimax-h3-ref2va-nv-origin-model
  PRIVATE ${BUDDY_SOURCE}/llvm/build/lib
)
target_link_libraries(minimax-h3-ref2va-nv-origin-model
  PRIVATE nv-runtime
          mlir_c_runner_utils
          mlir_runner_utils
          mlir_async_runtime
          mlir_cuda_runtime
          dl
          pthread
)

# ===----------------------------------------------------------------------===
# Build the Buddy InferenceRunner plugin.
# ===----------------------------------------------------------------------===
add_library(minimax-h3-ref2va-nv-origin-runner SHARED EXCLUDE_FROM_ALL runner.cpp)
set_target_properties(minimax-h3-ref2va-nv-origin-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_include_directories(minimax-h3-ref2va-nv-origin-runner
  PRIVATE ${LIB_DIR}
          ${OUTPUT_DIR}
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/runtime/include
          ${COMPILER_BUILD}/runtime/include
)
target_link_directories(minimax-h3-ref2va-nv-origin-runner
  PRIVATE ${BUDDY_SOURCE}/llvm/build/lib
)
target_link_libraries(minimax-h3-ref2va-nv-origin-runner
  PRIVATE dl
)

# ===----------------------------------------------------------------------===
# Package the RAX
#
# Contents:
# model library (libminimax-h3-ref2va-nv-origin-model.so),
# runner library (libminimax-h3-ref2va-nv-origin-runner.so),
# FP32 text encoder parameters (arg0_text_encoder.data),
# FP32 transformer parameters (arg0_transformer.data),
# FP32 visual decoder parameters (arg0_visual_vae.data),
# FP32 audio decoder parameters (arg0_audio_vae.data),
# remote-service helper (h3_api.py),
# FP32 visual encoder parameters (arg0_visual_vae_encode.data),
# MLIR runtime libraries (libmlir_c_runner_utils.so, libmlir_runner_utils.so,
# libmlir_async_runtime.so, libmlir_cuda_runtime.so).
#
# Output: output/minimax-h3-ref2va.rax embeds these files alongside RHAL metadata that
# identifies the runner, model library, parameters and runtime dependencies.
#
# ===----------------------------------------------------------------------===
configure_file(minimax-h3-ref2va.rhal.mlir.in ${OUTPUT_DIR}/minimax-h3-ref2va.rhal.mlir @ONLY)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/minimax-h3-ref2va.rax
  COMMAND ${COMPILER_BUILD}/bin/rax-pack ${OUTPUT_DIR}/minimax-h3-ref2va.rhal.mlir
          -o ${OUTPUT_DIR}/minimax-h3-ref2va.rax
          --embed-payload
  DEPENDS ${OUTPUT_DIR}/minimax-h3-ref2va.rhal.mlir
          minimax-h3-ref2va-nv-origin-model
          minimax-h3-ref2va-nv-origin-runner
          ${COMPILER_BUILD}/bin/rax-pack
          ${OUTPUT_DIR}/arg0_text_encoder.data
          ${OUTPUT_DIR}/arg0_transformer.data
          ${OUTPUT_DIR}/arg0_visual_vae.data
          ${OUTPUT_DIR}/arg0_audio_vae.data
          ${CMAKE_CURRENT_SOURCE_DIR}/h3_api.py
          ${OUTPUT_DIR}/arg0_visual_vae_encode.data
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_c_runner_utils.so
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_runner_utils.so
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_async_runtime.so
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_cuda_runtime.so
  COMMENT "Packaging NVIDIA minimax-h3-ref2va as RAX"
  VERBATIM
)
add_custom_target(buddy-nv-minimax-h3-ref2va-rax-origin-build DEPENDS ${OUTPUT_DIR}/minimax-h3-ref2va.rax)
