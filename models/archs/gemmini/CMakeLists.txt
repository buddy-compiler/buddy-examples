set(GEMMINI_LENET_DIR ${GEMMINI_DIR}/lenet)
set(GEMMINI_RESNET18_DIR ${GEMMINI_DIR}/resnet18)
set(GEMMINI_MOBILENETV3_DIR ${GEMMINI_DIR}/mobilenet-v3-small)
set(GEMMINI_BERT_DIR ${GEMMINI_DIR}/bert-base-uncased-emotion)
set(GEMMINI_STABLEDIFFUSION_DIR ${GEMMINI_DIR}/stable-diffusion)
set(GEMMINI_WHISPER_DIR ${GEMMINI_DIR}/whisper)
set(GEMMINI_DEEPSEEKR1_DIR ${GEMMINI_DIR}/deepseek-r1-distill-qwen-1.5b)
set(GEMMINI_QWEN3_DIR ${GEMMINI_DIR}/qwen3-8b)
set(GEMMINI_YOLO26_DIR ${GEMMINI_DIR}/yolo26n)
set(GEMMINI_GEMMA4_DIR ${GEMMINI_DIR}/gemma4-e2b-it)

if (MODEL_LENET)
  add_subdirectory(lenet)
endif()

if (MODEL_RESNET18)
  add_subdirectory(resnet18)
endif()

if (MODEL_MOBILENETV3)
  add_subdirectory(mobilenet-v3-small)
endif()

if (MODEL_BERT)
  add_subdirectory(bert-base-uncased-emotion)
endif()

if (MODEL_STABLEDIFFUSION)
  add_subdirectory(stable-diffusion)
endif()


if (MODEL_DEEPSEEKR1)
  add_subdirectory(deepseek-r1-distill-qwen-1.5b)
endif()

if (MODEL_QWEN3)
  add_subdirectory(qwen3-8b)
endif()

if (MODEL_YOLO26)
  add_subdirectory(yolo26n)
endif()


if (MODEL_GEMMA4)
  add_subdirectory(gemma4-e2b-it)
endif()

# add_custom_target(gemmini-model-build DEPENDS
#   GemminiLENET
#   GemminiRESNET18
#   GemminiMOBILENETV3
#   GemminiBERT
#   GemminiSTABLEDIFFUSION
#   GemminiDEEPSEEKR1
# )
