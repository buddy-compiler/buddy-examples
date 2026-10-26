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
          ${MODEL_DIR}/minimax-h3-fl2va/output/minimax-h3-fl2va/FL2VA/model_index.json
          ${CMAKE_CURRENT_SOURCE_DIR}/trace/audio_vae.toml
          ${CMAKE_CURRENT_SOURCE_DIR}/trace/text_encoder.toml
          ${CMAKE_CURRENT_SOURCE_DIR}/trace/transformer.toml
          ${CMAKE_CURRENT_SOURCE_DIR}/trace/visual_vae.toml
          ${CMAKE_CURRENT_SOURCE_DIR}/trace/visual_vae_encode.toml
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing minimax-h3-fl2va for x86"
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
# Compile the host entry point to native x86 code.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward_text_encoder.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_text_encoder_trace.mlir
             ${OUTPUT_DIR}/forward_text_encoder_linalg.mlir
             ${OUTPUT_DIR}/forward_text_encoder_llvm.mlir
             ${OUTPUT_DIR}/forward_text_encoder.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_text_encoder.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/forward_text_encoder_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_text_encoder_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/forward_text_encoder_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_text_encoder_linalg.mlir
            -convert-elementwise-to-linalg
            -convert-linalg-to-loops
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/forward_text_encoder_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/forward_text_encoder_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/forward_text_encoder.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward_text_encoder.ll -filetype=obj -relocation-model=pic -O3 -o forward_text_encoder.o
  DEPENDS ${OUTPUT_DIR}/forward_text_encoder.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling minimax-h3-fl2va forward_text_encoder for x86"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower model computation to native x86 code.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0_text_encoder.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_text_encoder_trace.mlir
             ${OUTPUT_DIR}/subgraph0_text_encoder_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_text_encoder_llvm.mlir
             ${OUTPUT_DIR}/subgraph0_text_encoder.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/subgraph0_text_encoder_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_text_encoder_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder_linalg.mlir
            -convert-elementwise-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-linalg-to-loops
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/subgraph0_text_encoder_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_text_encoder_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/subgraph0_text_encoder.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0_text_encoder.ll -filetype=obj -relocation-model=pic -O3 -o subgraph0_text_encoder.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_text_encoder.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling minimax-h3-fl2va subgraph0_text_encoder for x86"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Compile the host entry point to native x86 code.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward_transformer.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_transformer_trace.mlir
             ${OUTPUT_DIR}/forward_transformer_linalg.mlir
             ${OUTPUT_DIR}/forward_transformer_llvm.mlir
             ${OUTPUT_DIR}/forward_transformer.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_transformer.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/forward_transformer_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_transformer_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/forward_transformer_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_transformer_linalg.mlir
            -convert-elementwise-to-linalg
            -convert-linalg-to-loops
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/forward_transformer_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/forward_transformer_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/forward_transformer.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward_transformer.ll -filetype=obj -relocation-model=pic -O3 -o forward_transformer.o
  DEPENDS ${OUTPUT_DIR}/forward_transformer.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling minimax-h3-fl2va forward_transformer for x86"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower model computation to native x86 code.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0_transformer.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_transformer_trace.mlir
             ${OUTPUT_DIR}/subgraph0_transformer_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_transformer_llvm.mlir
             ${OUTPUT_DIR}/subgraph0_transformer.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_transformer.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/subgraph0_transformer_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_transformer_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_transformer_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_transformer_linalg.mlir
            -convert-elementwise-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-linalg-to-loops
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/subgraph0_transformer_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_transformer_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/subgraph0_transformer.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0_transformer.ll -filetype=obj -relocation-model=pic -O3 -o subgraph0_transformer.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_transformer.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling minimax-h3-fl2va subgraph0_transformer for x86"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Compile the host entry point to native x86 code.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward_visual_vae.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_visual_vae_trace.mlir
             ${OUTPUT_DIR}/forward_visual_vae_linalg.mlir
             ${OUTPUT_DIR}/forward_visual_vae_llvm.mlir
             ${OUTPUT_DIR}/forward_visual_vae.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_visual_vae.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/forward_visual_vae_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_visual_vae_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/forward_visual_vae_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_visual_vae_linalg.mlir
            -convert-elementwise-to-linalg
            -convert-linalg-to-loops
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/forward_visual_vae_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/forward_visual_vae_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/forward_visual_vae.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward_visual_vae.ll -filetype=obj -relocation-model=pic -O3 -o forward_visual_vae.o
  DEPENDS ${OUTPUT_DIR}/forward_visual_vae.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling minimax-h3-fl2va forward_visual_vae for x86"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower model computation to native x86 code.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0_visual_vae.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_visual_vae_trace.mlir
             ${OUTPUT_DIR}/subgraph0_visual_vae_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_visual_vae_llvm.mlir
             ${OUTPUT_DIR}/subgraph0_visual_vae.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_visual_vae.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/subgraph0_visual_vae_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_visual_vae_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_visual_vae_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_visual_vae_linalg.mlir
            -convert-elementwise-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-linalg-to-loops
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/subgraph0_visual_vae_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_visual_vae_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/subgraph0_visual_vae.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0_visual_vae.ll -filetype=obj -relocation-model=pic -O3 -o subgraph0_visual_vae.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_visual_vae.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling minimax-h3-fl2va subgraph0_visual_vae for x86"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Compile the host entry point to native x86 code.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward_visual_vae_encode.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_visual_vae_encode_trace.mlir
             ${OUTPUT_DIR}/forward_visual_vae_encode_linalg.mlir
             ${OUTPUT_DIR}/forward_visual_vae_encode_llvm.mlir
             ${OUTPUT_DIR}/forward_visual_vae_encode.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_visual_vae_encode.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/forward_visual_vae_encode_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_visual_vae_encode_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/forward_visual_vae_encode_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_visual_vae_encode_linalg.mlir
            -convert-elementwise-to-linalg
            -convert-linalg-to-loops
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/forward_visual_vae_encode_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/forward_visual_vae_encode_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/forward_visual_vae_encode.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward_visual_vae_encode.ll -filetype=obj -relocation-model=pic -O3 -o forward_visual_vae_encode.o
  DEPENDS ${OUTPUT_DIR}/forward_visual_vae_encode.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling minimax-h3-fl2va forward_visual_vae_encode for x86"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower model computation to native x86 code.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0_visual_vae_encode.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_visual_vae_encode_trace.mlir
             ${OUTPUT_DIR}/subgraph0_visual_vae_encode_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_visual_vae_encode_llvm.mlir
             ${OUTPUT_DIR}/subgraph0_visual_vae_encode.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_visual_vae_encode.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/subgraph0_visual_vae_encode_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_visual_vae_encode_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_visual_vae_encode_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_visual_vae_encode_linalg.mlir
            -convert-elementwise-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-linalg-to-loops
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/subgraph0_visual_vae_encode_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_visual_vae_encode_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/subgraph0_visual_vae_encode.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0_visual_vae_encode.ll -filetype=obj -relocation-model=pic -O3 -o subgraph0_visual_vae_encode.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_visual_vae_encode.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling minimax-h3-fl2va subgraph0_visual_vae_encode for x86"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Compile the host entry point to native x86 code.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward_audio_vae.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_audio_vae_trace.mlir
             ${OUTPUT_DIR}/forward_audio_vae_linalg.mlir
             ${OUTPUT_DIR}/forward_audio_vae_llvm.mlir
             ${OUTPUT_DIR}/forward_audio_vae.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_audio_vae.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/forward_audio_vae_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_audio_vae_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/forward_audio_vae_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_audio_vae_linalg.mlir
            -convert-elementwise-to-linalg
            -convert-linalg-to-loops
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/forward_audio_vae_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/forward_audio_vae_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/forward_audio_vae.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward_audio_vae.ll -filetype=obj -relocation-model=pic -O3 -o forward_audio_vae.o
  DEPENDS ${OUTPUT_DIR}/forward_audio_vae.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling minimax-h3-fl2va forward_audio_vae for x86"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower model computation to native x86 code.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0_audio_vae.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_audio_vae_trace.mlir
             ${OUTPUT_DIR}/subgraph0_audio_vae_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_audio_vae_llvm.mlir
             ${OUTPUT_DIR}/subgraph0_audio_vae.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_audio_vae.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/subgraph0_audio_vae_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_audio_vae_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_audio_vae_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_audio_vae_linalg.mlir
            -convert-elementwise-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-linalg-to-loops
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/subgraph0_audio_vae_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_audio_vae_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/subgraph0_audio_vae.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0_audio_vae.ll -filetype=obj -relocation-model=pic -O3 -o subgraph0_audio_vae.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_audio_vae.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling minimax-h3-fl2va subgraph0_audio_vae for x86"
  VERBATIM
)

add_library(minimax-h3-fl2va-x86-origin-model SHARED EXCLUDE_FROM_ALL forward_text_encoder.o subgraph0_text_encoder.o
  forward_transformer.o subgraph0_transformer.o forward_visual_vae.o subgraph0_visual_vae.o
  forward_visual_vae_encode.o subgraph0_visual_vae_encode.o forward_audio_vae.o subgraph0_audio_vae.o
)
set_target_properties(minimax-h3-fl2va-x86-origin-model
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             LINKER_LANGUAGE CXX
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_link_directories(minimax-h3-fl2va-x86-origin-model
  PRIVATE ${BUDDY_SOURCE}/llvm/build/lib
)
target_link_libraries(minimax-h3-fl2va-x86-origin-model
  PRIVATE x86-runtime
          mlir_c_runner_utils
          mlir_runner_utils
          dl
          pthread
)

# ===----------------------------------------------------------------------===
# Build the Buddy InferenceRunner plugin.
# ===----------------------------------------------------------------------===
add_library(minimax-h3-fl2va-x86-origin-runner SHARED EXCLUDE_FROM_ALL runner.cpp)
set_target_properties(minimax-h3-fl2va-x86-origin-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_include_directories(minimax-h3-fl2va-x86-origin-runner
  PRIVATE ${LIB_DIR}
          ${OUTPUT_DIR}
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/runtime/include
          ${COMPILER_BUILD}/runtime/include
)
target_link_directories(minimax-h3-fl2va-x86-origin-runner
  PRIVATE ${BUDDY_SOURCE}/llvm/build/lib
)
target_link_libraries(minimax-h3-fl2va-x86-origin-runner
  PRIVATE dl
)

# ===----------------------------------------------------------------------===
# Package the RAX
#
# Contents:
# model library (libminimax-h3-fl2va-x86-origin-model.so),
# runner library (libminimax-h3-fl2va-x86-origin-runner.so),
# FP32 text encoder parameters (arg0_text_encoder.data),
# FP32 transformer parameters (arg0_transformer.data),
# FP32 visual decoder parameters (arg0_visual_vae.data),
# FP32 audio decoder parameters (arg0_audio_vae.data),
# remote-service helper (h3_api.py),
# FP32 visual encoder parameters (arg0_visual_vae_encode.data),
# MLIR runtime libraries (libmlir_c_runner_utils.so, libmlir_runner_utils.so).
#
# Output: output/minimax-h3-fl2va.rax embeds these files alongside RHAL metadata that
# identifies the runner, model library, parameters and runtime dependencies.
#
# ===----------------------------------------------------------------------===
configure_file(minimax-h3-fl2va.rhal.mlir.in ${OUTPUT_DIR}/minimax-h3-fl2va.rhal.mlir @ONLY)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/minimax-h3-fl2va.rax
  COMMAND ${COMPILER_BUILD}/bin/rax-pack ${OUTPUT_DIR}/minimax-h3-fl2va.rhal.mlir
          -o ${OUTPUT_DIR}/minimax-h3-fl2va.rax
          --embed-payload
  DEPENDS ${OUTPUT_DIR}/minimax-h3-fl2va.rhal.mlir
          minimax-h3-fl2va-x86-origin-model
          minimax-h3-fl2va-x86-origin-runner
          ${COMPILER_BUILD}/bin/rax-pack
          ${OUTPUT_DIR}/arg0_text_encoder.data
          ${OUTPUT_DIR}/arg0_transformer.data
          ${OUTPUT_DIR}/arg0_visual_vae.data
          ${OUTPUT_DIR}/arg0_audio_vae.data
          ${CMAKE_CURRENT_SOURCE_DIR}/h3_api.py
          ${OUTPUT_DIR}/arg0_visual_vae_encode.data
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_c_runner_utils.so
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_runner_utils.so
  COMMENT "Packaging x86 minimax-h3-fl2va as RAX"
  VERBATIM
)
add_custom_target(buddy-x86-minimax-h3-fl2va-rax-origin-build DEPENDS ${OUTPUT_DIR}/minimax-h3-fl2va.rax)
