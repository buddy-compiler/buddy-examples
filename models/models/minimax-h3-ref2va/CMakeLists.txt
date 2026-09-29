set(DURATION_S 4 CACHE STRING "duration s")
set(FPS 24 CACHE STRING "fps")
set(HEIGHT 768 CACHE STRING "height")
set(WIDTH 1344 CACHE STRING "width")
set(LATENT_C 24 CACHE STRING "latent c")
set(AUDIO_LATENT_C 32 CACHE STRING "audio latent c")
set(TEXT_DIM 5120 CACHE STRING "text dim")
set(TEXT_LEN 512 CACHE STRING "text len")
set(SAMPLE_RATE 32000 CACHE STRING "sample rate")
set(DENOISE_STEPS 30 CACHE STRING "denoise steps")
set(VIDEO_SHIFT 12.0 CACHE STRING "video shift")
math(EXPR DEFAULT_FRAMES "${DURATION_S} * ${FPS}")
set(FRAMES ${DEFAULT_FRAMES} CACHE STRING "frames")
math(EXPR DEFAULT_LATENT_T "${FRAMES} / 4")
set(LATENT_T ${DEFAULT_LATENT_T} CACHE STRING "latent t")
math(EXPR DEFAULT_LATENT_H "${HEIGHT} / 16")
set(LATENT_H ${DEFAULT_LATENT_H} CACHE STRING "latent h")
math(EXPR DEFAULT_LATENT_W "${WIDTH} / 16")
set(LATENT_W ${DEFAULT_LATENT_W} CACHE STRING "latent w")
math(EXPR DEFAULT_AUDIO_LATENT_T "${DURATION_S} * 40")
set(AUDIO_LATENT_T ${DEFAULT_AUDIO_LATENT_T} CACHE STRING "audio latent t")
set(PARAMETER_DIR "${CMAKE_BINARY_DIR}/model-parameters/minimax-h3-ref2va")
file(MAKE_DIRECTORY "${PARAMETER_DIR}")
set(PARAMETER_HEADER "#pragma once\nnamespace model_parameters {\n")
set(MODEL_ARGUMENTS)
if(NOT "${DURATION_S}" MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "DURATION_S must be a positive integer")
endif()
string(APPEND PARAMETER_HEADER "inline constexpr int DURATION_S = ${DURATION_S};\n")
list(APPEND MODEL_ARGUMENTS --duration-s "${DURATION_S}")
if(NOT "${FPS}" MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "FPS must be a positive integer")
endif()
string(APPEND PARAMETER_HEADER "inline constexpr int FPS = ${FPS};\n")
list(APPEND MODEL_ARGUMENTS --fps "${FPS}")
if(NOT "${HEIGHT}" MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "HEIGHT must be a positive integer")
endif()
string(APPEND PARAMETER_HEADER "inline constexpr int HEIGHT = ${HEIGHT};\n")
list(APPEND MODEL_ARGUMENTS --height "${HEIGHT}")
if(NOT "${WIDTH}" MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "WIDTH must be a positive integer")
endif()
string(APPEND PARAMETER_HEADER "inline constexpr int WIDTH = ${WIDTH};\n")
list(APPEND MODEL_ARGUMENTS --width "${WIDTH}")
if(NOT "${LATENT_C}" MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "LATENT_C must be a positive integer")
endif()
string(APPEND PARAMETER_HEADER "inline constexpr int LATENT_C = ${LATENT_C};\n")
list(APPEND MODEL_ARGUMENTS --latent-c "${LATENT_C}")
if(NOT "${AUDIO_LATENT_C}" MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "AUDIO_LATENT_C must be a positive integer")
endif()
string(APPEND PARAMETER_HEADER "inline constexpr int AUDIO_LATENT_C = ${AUDIO_LATENT_C};\n")
list(APPEND MODEL_ARGUMENTS --audio-latent-c "${AUDIO_LATENT_C}")
if(NOT "${TEXT_DIM}" MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "TEXT_DIM must be a positive integer")
endif()
string(APPEND PARAMETER_HEADER "inline constexpr int TEXT_DIM = ${TEXT_DIM};\n")
list(APPEND MODEL_ARGUMENTS --text-dim "${TEXT_DIM}")
if(NOT "${TEXT_LEN}" MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "TEXT_LEN must be a positive integer")
endif()
string(APPEND PARAMETER_HEADER "inline constexpr int TEXT_LEN = ${TEXT_LEN};\n")
list(APPEND MODEL_ARGUMENTS --text-len "${TEXT_LEN}")
if(NOT "${SAMPLE_RATE}" MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "SAMPLE_RATE must be a positive integer")
endif()
string(APPEND PARAMETER_HEADER "inline constexpr int SAMPLE_RATE = ${SAMPLE_RATE};\n")
list(APPEND MODEL_ARGUMENTS --sample-rate "${SAMPLE_RATE}")
if(NOT "${DENOISE_STEPS}" MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "DENOISE_STEPS must be a positive integer")
endif()
string(APPEND PARAMETER_HEADER "inline constexpr int DENOISE_STEPS = ${DENOISE_STEPS};\n")
list(APPEND MODEL_ARGUMENTS --denoise-steps "${DENOISE_STEPS}")
string(APPEND PARAMETER_HEADER "inline constexpr float VIDEO_SHIFT = static_cast<float>(${VIDEO_SHIFT});\n")
list(APPEND MODEL_ARGUMENTS --video-shift "${VIDEO_SHIFT}")
if(NOT "${FRAMES}" MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "FRAMES must be a positive integer")
endif()
string(APPEND PARAMETER_HEADER "inline constexpr int FRAMES = ${FRAMES};\n")
list(APPEND MODEL_ARGUMENTS --frames "${FRAMES}")
if(NOT "${LATENT_T}" MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "LATENT_T must be a positive integer")
endif()
string(APPEND PARAMETER_HEADER "inline constexpr int LATENT_T = ${LATENT_T};\n")
list(APPEND MODEL_ARGUMENTS --latent-t "${LATENT_T}")
if(NOT "${LATENT_H}" MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "LATENT_H must be a positive integer")
endif()
string(APPEND PARAMETER_HEADER "inline constexpr int LATENT_H = ${LATENT_H};\n")
list(APPEND MODEL_ARGUMENTS --latent-h "${LATENT_H}")
if(NOT "${LATENT_W}" MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "LATENT_W must be a positive integer")
endif()
string(APPEND PARAMETER_HEADER "inline constexpr int LATENT_W = ${LATENT_W};\n")
list(APPEND MODEL_ARGUMENTS --latent-w "${LATENT_W}")
if(NOT "${AUDIO_LATENT_T}" MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "AUDIO_LATENT_T must be a positive integer")
endif()
string(APPEND PARAMETER_HEADER "inline constexpr int AUDIO_LATENT_T = ${AUDIO_LATENT_T};\n")
list(APPEND MODEL_ARGUMENTS --audio-latent-t "${AUDIO_LATENT_T}")
string(APPEND PARAMETER_HEADER "}\n")
file(CONFIGURE OUTPUT "${PARAMETER_DIR}/model_parameters.h" CONTENT "${PARAMETER_HEADER}")
add_custom_command(
  OUTPUT ${MODEL_MINIMAX_H3_REF2VA_DIR}/arg0_text_encoder.data
         ${MODEL_MINIMAX_H3_REF2VA_DIR}/arg0_transformer.data
         ${MODEL_MINIMAX_H3_REF2VA_DIR}/arg0_visual_vae.data
         ${MODEL_MINIMAX_H3_REF2VA_DIR}/arg0_audio_vae.data
         ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_text_encoder.mlir
         ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_text_encoder.mlir
         ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_transformer.mlir
         ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_transformer.mlir
         ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_visual_vae.mlir
         ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_visual_vae.mlir
         ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_visual_vae_encode.mlir
         ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_visual_vae_encode.mlir
         ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_audio_vae.mlir
         ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_audio_vae.mlir
  COMMAND ${Python3_EXECUTABLE} ${MODEL_MINIMAX_H3_REF2VA_DIR}/import-minimax-h3-ref2va.py
          --output-dir ${MODEL_MINIMAX_H3_REF2VA_DIR}
          ${MODEL_ARGUMENTS}
          --trace
  DEPENDS ${PARAMETER_DIR}/model_parameters.h
          ${CMAKE_CURRENT_SOURCE_DIR}/import-minimax-h3-ref2va.py
          ${CMAKE_CURRENT_SOURCE_DIR}/configs/importer-param.py
  COMMENT "Generating MiniMax-H3 Ref2VA MLIR and parameter files"
)

add_custom_target(minimax-h3-ref2va-model-build DEPENDS
  ${MODEL_MINIMAX_H3_REF2VA_DIR}/arg0_text_encoder.data
  ${MODEL_MINIMAX_H3_REF2VA_DIR}/arg0_transformer.data
  ${MODEL_MINIMAX_H3_REF2VA_DIR}/arg0_visual_vae.data
  ${MODEL_MINIMAX_H3_REF2VA_DIR}/arg0_audio_vae.data
  ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_text_encoder.mlir
  ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_text_encoder.mlir
  ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_transformer.mlir
  ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_transformer.mlir
  ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_visual_vae.mlir
  ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_visual_vae.mlir
  ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_visual_vae_encode.mlir
  ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_visual_vae_encode.mlir
  ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_audio_vae.mlir
  ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_audio_vae.mlir
)
