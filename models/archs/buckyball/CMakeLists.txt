# Select exactly one Buckyball chip layout for this workload build.
if(DEFINED BUCKYBALL_WORKLOAD_CHIP)
  set(BUCKYBALL_ARCH "${BUCKYBALL_WORKLOAD_CHIP}")
else()
  set(BUCKYBALL_ARCH "" CACHE STRING
      "Buckyball chip layout: pebble, goban, or poly")
endif()
if(NOT BUCKYBALL_ARCH MATCHES "^(pebble|goban|poly)$")
  message(FATAL_ERROR
    "BUCKYBALL_ARCH must be pebble, goban, or poly; got '${BUCKYBALL_ARCH}'")
endif()

set(BUCKYBALL_DIR ${CMAKE_CURRENT_LIST_DIR}/${BUCKYBALL_ARCH})
set(BUCKYBALL_OUTPUT_DIR
    ${OUTPUT_BIN_DIR}/src/ModelTest/e2e/models/archs/buckyball/${BUCKYBALL_ARCH})
set(BUCKYBALL_LENET_DIR ${BUCKYBALL_DIR}/LeNet)
set(BUCKYBALL_RESNET18_DIR ${BUCKYBALL_DIR}/ResNet18)
set(BUCKYBALL_MOBILENETV3_DIR ${BUCKYBALL_DIR}/MobileNetV3)
set(BUCKYBALL_BERT_DIR ${BUCKYBALL_DIR}/Bert)
set(BUCKYBALL_STABLEDIFFUSION_DIR ${BUCKYBALL_DIR}/StableDiffusion)
set(BUCKYBALL_LLAMA2_DIR ${BUCKYBALL_DIR}/llama2)
set(BUCKYBALL_DEEPSEEKR1_DIR ${BUCKYBALL_DIR}/DeepSeekR1)
set(BUCKYBALL_QWEN3_DIR ${BUCKYBALL_DIR}/Qwen3)
set(BUCKYBALL_YOLO26_DIR ${BUCKYBALL_DIR}/YOLO26)
set(BUCKYBALL_BUDDYNEXT_DIR ${BUCKYBALL_DIR}/BuddyNext)
set(BUCKYBALL_GEMMA4_DIR ${BUCKYBALL_DIR}/Gemma4)
set(BUCKYBALL_WHISPER_DIR ${BUCKYBALL_DIR}/Whisper)

set(BUCKYBALL_PEBBLE_MODELS
    LeNet ResNet18 MobileNetV3 YOLO26 BuddyNext DeepSeekR1 StableDiffusion
    Whisper llama2)
set(BUCKYBALL_GOBAN_MODELS Bert)
set(BUCKYBALL_POLY_MODELS Gemma4 Qwen3)
string(TOUPPER "${BUCKYBALL_ARCH}" BUCKYBALL_ARCH_UPPER)
set(BUCKYBALL_ALLOWED_MODELS
    ${BUCKYBALL_${BUCKYBALL_ARCH_UPPER}_MODELS})

set(BUCKYBALL_ALL_MODELS
    LENET RESNET18 MOBILENETV3 YOLO26 BUDDYNEXT DEEPSEEKR1 STABLEDIFFUSION
    WHISPER LLAMA2 BERT GEMMA4 QWEN3 MINIMAX_H3_FL2VA MINIMAX_H3_REF2VA)
foreach(model_upper IN LISTS BUCKYBALL_ALL_MODELS)
  if(MODEL_${model_upper})
    set(model_allowed FALSE)
    foreach(allowed_model IN LISTS BUCKYBALL_ALLOWED_MODELS)
      string(TOUPPER "${allowed_model}" allowed_model_upper)
      if(allowed_model_upper STREQUAL model_upper)
        set(model_allowed TRUE)
      endif()
    endforeach()
    if(NOT model_allowed)
      message(FATAL_ERROR
        "Model '${model_upper}' is not supported by Buckyball arch '${BUCKYBALL_ARCH}'")
    endif()
  endif()
endforeach()

include(${CMAKE_CURRENT_LIST_DIR}/RushB.cmake)

function(add_buckyball_perfetto_target target)
  set(one_value_args TRACE_DIR TRACE_TOML OUTPUT)
  set(multi_value_args MLIR)
  cmake_parse_arguments(PFT "" "${one_value_args}" "${multi_value_args}" ${ARGN})
  if(NOT PFT_TRACE_DIR OR NOT PFT_TRACE_TOML OR NOT PFT_MLIR)
    message(FATAL_ERROR "${target} requires TRACE_DIR, TRACE_TOML, and MLIR")
  endif()
  if(PFT_OUTPUT)
    set(output ${PFT_OUTPUT})
  else()
    set(output ${PFT_TRACE_DIR}/perfetto.json)
  endif()
  set(mlir_args)
  foreach(mlir IN LISTS PFT_MLIR)
    list(APPEND mlir_args --mlir ${mlir})
  endforeach()
  add_custom_target(${target}
    COMMAND ${Python3_EXECUTABLE}
            ${WORKLOAD_DIR}/../framework/trace/perfetto.py
            ${PFT_TRACE_DIR} ${PFT_TRACE_TOML} -o ${output} ${mlir_args}
    COMMENT "Generating Perfetto trace ${output}"
    VERBATIM USES_TERMINAL)
endfunction()

if(MODEL_LENET AND BUCKYBALL_ARCH STREQUAL "pebble")
  add_subdirectory(pebble/LeNet)
  add_buckyball_rushb_targets(LeNet buddy-buckyball-lenet ${MODEL_DIR}/LeNet
    forward.mlir subgraph0.mlir)
endif()
if(MODEL_RESNET18 AND BUCKYBALL_ARCH STREQUAL "pebble")
  add_subdirectory(pebble/ResNet18)
  add_buckyball_rushb_targets(ResNet18 buddy-buckyball-resnet ${MODEL_DIR}/ResNet18
    forward.mlir subgraph0.mlir)
endif()
if(MODEL_MOBILENETV3 AND BUCKYBALL_ARCH STREQUAL "pebble")
  add_subdirectory(pebble/MobileNetV3)
  add_buckyball_rushb_targets(MobileNetV3 buddy-buckyball-mobilenetv3
    ${MODEL_DIR}/MobileNetV3 forward.mlir subgraph0.mlir)
endif()
if(MODEL_YOLO26 AND BUCKYBALL_ARCH STREQUAL "pebble")
  add_subdirectory(pebble/YOLO26)
  add_buckyball_rushb_targets(YOLO26 buddy-buckyball-yolo26 ${MODEL_DIR}/YOLO26
    forward.mlir subgraph0.mlir)
endif()
if(MODEL_BUDDYNEXT AND BUCKYBALL_ARCH STREQUAL "pebble")
  add_subdirectory(pebble/BuddyNext)
  file(GLOB BUDDYNEXT_RUSHB_MLIR CONFIGURE_DEPENDS
    ${MODEL_DIR}/BuddyNext/prefill/*.mlir ${MODEL_DIR}/BuddyNext/decode/*.mlir)
  add_buckyball_rushb_targets(BuddyNext buddy-buckyball-buddynext
    ${MODEL_DIR}/BuddyNext ${BUDDYNEXT_RUSHB_MLIR})
endif()
if(MODEL_DEEPSEEKR1 AND BUCKYBALL_ARCH STREQUAL "pebble")
  add_subdirectory(pebble/DeepSeekR1)
  add_buckyball_rushb_targets(DeepSeekR1 buddy-buckyball-deepseekr1
    ${MODEL_DIR}/DeepSeekR1 forward.mlir subgraph0.mlir)
endif()
if(MODEL_STABLEDIFFUSION AND BUCKYBALL_ARCH STREQUAL "pebble")
  add_subdirectory(pebble/StableDiffusion)
  add_buckyball_rushb_targets(StableDiffusion buddy-buckyball-stable-diffusion
    ${MODEL_DIR}/StableDiffusion forward_text_encoder.mlir
    subgraph0_text_encoder.mlir forward_unet.mlir subgraph0_unet.mlir
    forward_vae.mlir subgraph0_vae.mlir)
endif()
if(MODEL_WHISPER AND BUCKYBALL_ARCH STREQUAL "pebble")
  add_subdirectory(pebble/Whisper)
  add_buckyball_rushb_targets(Whisper buddy-buckyball-whisper
    ${CMAKE_BINARY_DIR}/workloads/src/ModelTest/e2e/models/models/Whisper
    forward.mlir subgraph0.mlir)
endif()
if(MODEL_LLAMA2 AND BUCKYBALL_ARCH STREQUAL "pebble")
  add_subdirectory(pebble/llama2)
  add_buckyball_rushb_targets(llama2 buddy-buckyball-llama2 ${MODEL_DIR}/llama2
    forward.mlir subgraph0.mlir)
endif()
if(MODEL_BERT AND BUCKYBALL_ARCH STREQUAL "goban")
  add_subdirectory(goban/Bert)
  add_buckyball_rushb_targets(Bert buddy-buckyball-bert ${MODEL_DIR}/Bert
    forward.mlir subgraph0.mlir subgraph1.mlir subgraph2.mlir subgraph3.mlir)
endif()
if(MODEL_GEMMA4 AND BUCKYBALL_ARCH STREQUAL "poly")
  add_subdirectory(poly/Gemma4)
  add_buckyball_rushb_targets(Gemma4 buddy-buckyball-gemma4 ${MODEL_DIR}/Gemma4
    forward_prefill.mlir subgraph0_prefill.mlir
    forward_decode.mlir subgraph0_decode.mlir)
endif()
if(MODEL_QWEN3 AND BUCKYBALL_ARCH STREQUAL "poly")
  add_subdirectory(poly/Qwen3)
  add_buckyball_rushb_targets(Qwen3 buddy-buckyball-qwen3 ${MODEL_DIR}/Qwen3
    forward_prefill_0_6b.mlir subgraph0_prefill_0_6b.mlir
    forward_decode_0_6b.mlir subgraph0_decode_0_6b.mlir)
  add_dependencies(buddy-buckyball-qwen3-rushB-codegen qwen3-model-build)
endif()
