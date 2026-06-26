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

function(add_buckyball_perfetto_target target)
  set(one_value_args TRACE_DIR TRACE_TOML OUTPUT)
  set(multi_value_args MLIR DEPENDS)
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
    DEPENDS ${PFT_DEPENDS} ${PFT_MLIR}
    COMMENT "Generating Perfetto trace ${output}"
    VERBATIM
    USES_TERMINAL
  )
endfunction()

if (MODEL_LENET)
  add_subdirectory(LeNet)
endif()

if (MODEL_RESNET18)
  add_subdirectory(ResNet18)
endif()

if (MODEL_MOBILENETV3)
  add_subdirectory(MobileNetV3)
endif()

if (MODEL_BERT)
  add_subdirectory(Bert)
endif()

if (MODEL_STABLEDIFFUSION)
  add_subdirectory(StableDiffusion)
endif()

if (MODEL_LLAMA2)
  add_subdirectory(llama2)
endif()

if (MODEL_DEEPSEEKR1)
  add_subdirectory(DeepSeekR1)
endif()

if (MODEL_QWEN3)
  add_subdirectory(Qwen3)
endif()

if (MODEL_YOLO26)
  add_subdirectory(YOLO26)
endif()

if (MODEL_BUDDYNEXT)
  add_subdirectory(BuddyNext)
endif()

if (MODEL_GEMMA4)
  add_subdirectory(Gemma4)
endif()

if (MODEL_WHISPER)
  add_subdirectory(Whisper)
endif()
