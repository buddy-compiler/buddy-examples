# Select exactly one Buckyball chip layout for this workload build.
# Model↔chip is many-to-many: a chip may host many models, and the same
# model name may appear under multiple chips. Layout dirs under this folder
# are the source of truth for which pairs exist today.
if(DEFINED BUCKYBALL_WORKLOAD_CHIP)
  set(BUCKYBALL_ARCH "${BUCKYBALL_WORKLOAD_CHIP}")
else()
  set(BUCKYBALL_ARCH "" CACHE STRING
      "Buckyball chip layout directory under archs/buckyball/")
endif()

set(BUCKYBALL_DIR ${CMAKE_CURRENT_LIST_DIR}/${BUCKYBALL_ARCH})
if(BUCKYBALL_ARCH STREQUAL "" OR NOT EXISTS ${BUCKYBALL_DIR})
  message(FATAL_ERROR
    "No Buckyball ModelTest layout for chip '${BUCKYBALL_ARCH}' "
    "(expected directory ${CMAKE_CURRENT_LIST_DIR}/<chip>)")
endif()

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

file(GLOB BUCKYBALL_LAYOUT_ENTRIES LIST_DIRECTORIES true
  RELATIVE ${BUCKYBALL_DIR} ${BUCKYBALL_DIR}/*)
set(BUCKYBALL_ALLOWED_MODELS)
foreach(entry IN LISTS BUCKYBALL_LAYOUT_ENTRIES)
  if(IS_DIRECTORY ${BUCKYBALL_DIR}/${entry})
    list(APPEND BUCKYBALL_ALLOWED_MODELS ${entry})
  endif()
endforeach()

# Explicit -DMODEL= must match a layout on this chip. Default-all MODEL_*
# without -DMODEL= only builds layouts that exist (many-to-many; no 1:1 gate).
if(DEFINED MODEL)
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
          "Model '${model_upper}' is not supported by Buckyball chip '${BUCKYBALL_ARCH}' "
          "(layout dirs: ${BUCKYBALL_ALLOWED_MODELS})")
      endif()
    endif()
  endforeach()
endif()

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

if(MODEL_LENET AND EXISTS ${BUCKYBALL_DIR}/LeNet)
  add_subdirectory(${BUCKYBALL_ARCH}/LeNet)
  add_buckyball_rushb_targets(LeNet buddy-buckyball-lenet ${MODEL_DIR}/LeNet
    forward.mlir subgraph0.mlir)
endif()
if(MODEL_RESNET18 AND EXISTS ${BUCKYBALL_DIR}/ResNet18)
  add_subdirectory(${BUCKYBALL_ARCH}/ResNet18)
  add_buckyball_rushb_targets(ResNet18 buddy-buckyball-resnet ${MODEL_DIR}/ResNet18
    forward.mlir subgraph0.mlir)
endif()
if(MODEL_MOBILENETV3 AND EXISTS ${BUCKYBALL_DIR}/MobileNetV3)
  add_subdirectory(${BUCKYBALL_ARCH}/MobileNetV3)
  add_buckyball_rushb_targets(MobileNetV3 buddy-buckyball-mobilenetv3
    ${MODEL_DIR}/MobileNetV3 forward.mlir subgraph0.mlir)
endif()
if(MODEL_YOLO26 AND EXISTS ${BUCKYBALL_DIR}/YOLO26)
  add_subdirectory(${BUCKYBALL_ARCH}/YOLO26)
  add_buckyball_rushb_targets(YOLO26 buddy-buckyball-yolo26 ${MODEL_DIR}/YOLO26
    forward.mlir subgraph0.mlir)
endif()
if(MODEL_BUDDYNEXT AND EXISTS ${BUCKYBALL_DIR}/BuddyNext)
  add_subdirectory(${BUCKYBALL_ARCH}/BuddyNext)
  file(GLOB BUDDYNEXT_RUSHB_MLIR CONFIGURE_DEPENDS
    ${MODEL_DIR}/BuddyNext/prefill/*.mlir ${MODEL_DIR}/BuddyNext/decode/*.mlir)
  add_buckyball_rushb_targets(BuddyNext buddy-buckyball-buddynext
    ${MODEL_DIR}/BuddyNext ${BUDDYNEXT_RUSHB_MLIR})
endif()
if(MODEL_DEEPSEEKR1 AND EXISTS ${BUCKYBALL_DIR}/DeepSeekR1)
  add_subdirectory(${BUCKYBALL_ARCH}/DeepSeekR1)
  add_buckyball_rushb_targets(DeepSeekR1 buddy-buckyball-deepseekr1
    ${MODEL_DIR}/DeepSeekR1 forward.mlir subgraph0.mlir)
endif()
if(MODEL_STABLEDIFFUSION AND EXISTS ${BUCKYBALL_DIR}/StableDiffusion)
  add_subdirectory(${BUCKYBALL_ARCH}/StableDiffusion)
  add_buckyball_rushb_targets(StableDiffusion buddy-buckyball-stable-diffusion
    ${MODEL_DIR}/StableDiffusion forward_text_encoder.mlir
    subgraph0_text_encoder.mlir forward_unet.mlir subgraph0_unet.mlir
    forward_vae.mlir subgraph0_vae.mlir)
endif()
if(MODEL_WHISPER AND EXISTS ${BUCKYBALL_DIR}/Whisper)
  add_subdirectory(${BUCKYBALL_ARCH}/Whisper)
  add_buckyball_rushb_targets(Whisper buddy-buckyball-whisper
    ${CMAKE_BINARY_DIR}/workloads/src/ModelTest/e2e/models/models/Whisper
    forward.mlir subgraph0.mlir)
endif()
if(MODEL_LLAMA2 AND EXISTS ${BUCKYBALL_DIR}/llama2)
  add_subdirectory(${BUCKYBALL_ARCH}/llama2)
  add_buckyball_rushb_targets(llama2 buddy-buckyball-llama2 ${MODEL_DIR}/llama2
    forward.mlir subgraph0.mlir)
endif()
if(MODEL_BERT AND EXISTS ${BUCKYBALL_DIR}/Bert)
  add_subdirectory(${BUCKYBALL_ARCH}/Bert)
  add_buckyball_rushb_targets(Bert buddy-buckyball-bert ${MODEL_DIR}/Bert
    forward.mlir subgraph0.mlir subgraph1.mlir subgraph2.mlir subgraph3.mlir)
endif()
if(MODEL_GEMMA4 AND EXISTS ${BUCKYBALL_DIR}/Gemma4)
  add_subdirectory(${BUCKYBALL_ARCH}/Gemma4)
  add_buckyball_rushb_targets(Gemma4 buddy-buckyball-gemma4 ${MODEL_DIR}/Gemma4
    forward_prefill.mlir subgraph0_prefill.mlir
    forward_decode.mlir subgraph0_decode.mlir)
endif()
if(MODEL_QWEN3 AND EXISTS ${BUCKYBALL_DIR}/Qwen3)
  add_subdirectory(${BUCKYBALL_ARCH}/Qwen3)
  add_buckyball_rushb_targets(Qwen3 buddy-buckyball-qwen3 ${MODEL_DIR}/Qwen3
    forward_prefill_0_6b.mlir subgraph0_prefill_0_6b.mlir
    forward_decode_0_6b.mlir subgraph0_decode_0_6b.mlir)
  add_dependencies(buddy-buckyball-qwen3-rushB-codegen qwen3-model-build)
endif()
