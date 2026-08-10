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

include(${CMAKE_CURRENT_LIST_DIR}/RushB.cmake)

function(add_buckyball_perfetto_target target)
  set(one_value_args TRACE_DIR TRACE_TOML OUTPUT)
  set(multi_value_args MLIR)
  cmake_parse_arguments(PFT "" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if (NOT PFT_TRACE_DIR)
    message(FATAL_ERROR "${target} missing TRACE_DIR")
  endif()
  if (NOT PFT_TRACE_TOML)
    message(FATAL_ERROR "${target} missing TRACE_TOML")
  endif()
  if (NOT PFT_MLIR)
    message(FATAL_ERROR "${target} missing MLIR")
  endif()

  if (PFT_OUTPUT)
    set(output ${PFT_OUTPUT})
  else()
    set(output ${PFT_TRACE_DIR}/perfetto.json)
  endif()

  set(mlir_args)
  foreach(mlir ${PFT_MLIR})
    list(APPEND mlir_args --mlir ${mlir})
  endforeach()

  add_custom_target(${target}
    COMMAND ${Python3_EXECUTABLE}
            ${WORKLOAD_DIR}/../scripts/perfetto.py
            ${PFT_TRACE_DIR}
            ${PFT_TRACE_TOML}
            -o ${output}
            ${mlir_args}
    COMMENT "Generating Perfetto trace ${output}"
    VERBATIM
    USES_TERMINAL
  )
endfunction()

if (MODEL_LENET)
  add_subdirectory(LeNet)
  add_buckyball_rushb_targets(LeNet buddy-buckyball-lenet ${MODEL_DIR}/LeNet
    forward.mlir subgraph0.mlir)
endif()

if (MODEL_RESNET18)
  add_subdirectory(ResNet18)
  add_buckyball_rushb_targets(ResNet18 buddy-buckyball-resnet ${MODEL_DIR}/ResNet18
    forward.mlir subgraph0.mlir)
endif()

if (MODEL_MOBILENETV3)
  add_subdirectory(MobileNetV3)
  add_buckyball_rushb_targets(MobileNetV3 buddy-buckyball-mobilenetv3
    ${MODEL_DIR}/MobileNetV3 forward.mlir subgraph0.mlir)
endif()

if (MODEL_BERT)
  add_subdirectory(Bert)
  add_buckyball_rushb_targets(Bert buddy-buckyball-bert ${MODEL_DIR}/Bert
    forward.mlir subgraph0.mlir subgraph1.mlir subgraph2.mlir subgraph3.mlir)
endif()

if (MODEL_STABLEDIFFUSION)
  add_subdirectory(StableDiffusion)
  add_buckyball_rushb_targets(StableDiffusion buddy-buckyball-stable-diffusion
    ${MODEL_DIR}/StableDiffusion
    forward_text_encoder.mlir subgraph0_text_encoder.mlir
    forward_unet.mlir subgraph0_unet.mlir
    forward_vae.mlir subgraph0_vae.mlir)
endif()

if (MODEL_LLAMA2)
  add_subdirectory(llama2)
  add_buckyball_rushb_targets(llama2 buddy-buckyball-llama2 ${MODEL_DIR}/llama2
    forward.mlir subgraph0.mlir)
endif()

if (MODEL_DEEPSEEKR1)
  add_subdirectory(DeepSeekR1)
  add_buckyball_rushb_targets(DeepSeekR1 buddy-buckyball-deepseekr1
    ${MODEL_DIR}/DeepSeekR1 forward.mlir subgraph0.mlir)
endif()

if (MODEL_QWEN3)
  add_subdirectory(Qwen3)
  add_buckyball_rushb_targets(Qwen3 buddy-buckyball-qwen3 ${MODEL_DIR}/Qwen3
    forward_prefill_0_6b.mlir subgraph0_prefill_0_6b.mlir
    forward_decode_0_6b.mlir subgraph0_decode_0_6b.mlir)
  # The importer owns the generated prefill graph.  RushB codegen must wait
  # for it, otherwise a fresh checkout fails while copying a missing MLIR.
  add_dependencies(buddy-buckyball-qwen3-rushB-codegen qwen3-model-build)
endif()

if (MODEL_YOLO26)
  add_subdirectory(YOLO26)
  add_buckyball_rushb_targets(YOLO26 buddy-buckyball-yolo26 ${MODEL_DIR}/YOLO26
    forward.mlir subgraph0.mlir)
endif()

if (MODEL_BUDDYNEXT)
  add_subdirectory(BuddyNext)
  file(GLOB BUDDYNEXT_RUSHB_MLIR CONFIGURE_DEPENDS
    ${MODEL_DIR}/BuddyNext/prefill/*.mlir
    ${MODEL_DIR}/BuddyNext/decode/*.mlir)
  add_buckyball_rushb_targets(BuddyNext buddy-buckyball-buddynext
    ${MODEL_DIR}/BuddyNext ${BUDDYNEXT_RUSHB_MLIR})
endif()

if (MODEL_GEMMA4)
  add_subdirectory(Gemma4)
  add_buckyball_rushb_targets(Gemma4 buddy-buckyball-gemma4 ${MODEL_DIR}/Gemma4
    forward_prefill.mlir subgraph0_prefill.mlir
    forward_decode.mlir subgraph0_decode.mlir)
endif()

if (MODEL_WHISPER)
  add_subdirectory(Whisper)
  add_buckyball_rushb_targets(Whisper buddy-buckyball-whisper
    ${CMAKE_BINARY_DIR}/workloads/src/ModelTest/e2e/models/models/Whisper
    forward.mlir subgraph0.mlir)
endif()
