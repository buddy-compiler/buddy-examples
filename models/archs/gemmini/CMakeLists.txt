set(GEMMINI_LENET_DIR ${GEMMINI_DIR}/LeNet)
set(GEMMINI_RESNET18_DIR ${GEMMINI_DIR}/ResNet18)
set(GEMMINI_MOBILENETV3_DIR ${GEMMINI_DIR}/MobileNetV3)
set(GEMMINI_BERT_DIR ${GEMMINI_DIR}/Bert)
set(GEMMINI_STABLEDIFFUSION_DIR ${GEMMINI_DIR}/StableDiffusion)
set(GEMMINI_WHISPER_DIR ${GEMMINI_DIR}/Whisper)
set(GEMMINI_LLAMA2_DIR ${GEMMINI_DIR}/Llama2)
set(GEMMINI_DEEPSEEKR1_DIR ${GEMMINI_DIR}/DeepSeekR1)
set(GEMMINI_QWEN3_DIR ${GEMMINI_DIR}/Qwen3)
set(GEMMINI_YOLO26_DIR ${GEMMINI_DIR}/YOLO26)
set(GEMMINI_BUDDYNEXT_DIR ${GEMMINI_DIR}/BuddyNext)
set(GEMMINI_GEMMA4_DIR ${GEMMINI_DIR}/Gemma4)

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

# add_custom_target(gemmini-model-build DEPENDS
#   GemminiLENET
#   GemminiRESNET18
#   GemminiMOBILENETV3
#   GemminiBERT
#   GemminiSTABLEDIFFUSION
#   GemminiLLAMA2
#   GemminiDEEPSEEKR1
# )
