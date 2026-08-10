set(MODEL_DEEPSEEKR1_DIR ${MODEL_DIR}/DeepSeekR1)
set(MODEL_LENET_DIR ${MODEL_DIR}/LeNet)
set(MODEL_RESNET18_DIR ${MODEL_DIR}/ResNet18)
set(MODEL_BERT_DIR ${MODEL_DIR}/Bert)
set(MODEL_MOBILENETV3_DIR ${MODEL_DIR}/MobileNetV3)
set(MODEL_WHISPER_DIR ${MODEL_DIR}/Whisper)
set(MODEL_STABLEDIFFUSION_DIR ${MODEL_DIR}/StableDiffusion)
set(MODEL_LLAMA2_DIR ${MODEL_DIR}/llama2)
set(MODEL_QWEN3_DIR ${MODEL_DIR}/Qwen3)
set(MODEL_YOLO26_DIR ${MODEL_DIR}/YOLO26)
set(MODEL_BUDDYNEXT_DIR ${MODEL_DIR}/BuddyNext)
set(MODEL_GEMMA4_DIR ${MODEL_DIR}/Gemma4)
set(MODEL_MINIMAX_H3_FL2VA_DIR ${MODEL_DIR}/MiniMaxH3FL2VA)
set(MODEL_MINIMAX_H3_REF2VA_DIR ${MODEL_DIR}/MiniMaxH3Ref2VA)
set(MODEL_BGE_M3_DIR ${MODEL_DIR}/BgeM3)
set(MODEL_PROTEINGLM_DIR ${MODEL_DIR}/ProteinGLM)
set(MODEL_QWEN3VL_DIR ${MODEL_DIR}/Qwen3VL)
set(MODEL_TIMESFM_DIR ${MODEL_DIR}/timesfm)
set(MODEL_MISTRAL_DIR ${MODEL_DIR}/mistral)
set(MODEL_KOKORO_DIR ${MODEL_DIR}/kokoro)
set(MODEL_MOLFORMER_DIR ${MODEL_DIR}/molformer)
set(MODEL_COLBERTV2_DIR ${MODEL_DIR}/colbertv2)
set(MODEL_MMS_DIR ${MODEL_DIR}/mms)
set(MODEL_E5_MISTRAL_DIR ${MODEL_DIR}/e5-mistral)
set(MODEL_CHEMBERTA_DIR ${MODEL_DIR}/chemberta)
set(MODEL_SAM2_DIR ${MODEL_DIR}/sam2)
set(MODEL_SMOLVLM2_DIR ${MODEL_DIR}/smolvlm2)
set(MODEL_BGE_RERANKER_DIR ${MODEL_DIR}/bge-reranker)
set(MODEL_PALIGEMMA_DIR ${MODEL_DIR}/paligemma)

# Conditionally add model subdirectories based on MODEL_* flags
if (MODEL_LENET)
  add_subdirectory(LeNet)
endif()

if (MODEL_DEEPSEEKR1)
  add_subdirectory(DeepSeekR1)
endif()

if (MODEL_RESNET18)
  add_subdirectory(ResNet18)
endif()

if (MODEL_BERT)
  add_subdirectory(Bert)
endif()

if (MODEL_MOBILENETV3)
  add_subdirectory(MobileNetV3)
endif()

if (MODEL_WHISPER)
  add_subdirectory(Whisper)
endif()

if (MODEL_STABLEDIFFUSION)
  add_subdirectory(StableDiffusion)
endif()

if (MODEL_LLAMA2)
  add_subdirectory(llama2)
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

if (MODEL_MINIMAX_H3_FL2VA)
  add_subdirectory(MiniMaxH3FL2VA)
endif()

if (MODEL_MINIMAX_H3_REF2VA)
  add_subdirectory(MiniMaxH3Ref2VA)
endif()

# These models are imported through buddy-cli's RAX workflow rather than an
# accelerator backend in this repository. Keep their import targets available
# without assigning them to a Gemmini, Buckyball, or NV layout.
if (MODEL_BGE_M3)
  add_subdirectory(BgeM3)
endif()

if (MODEL_PROTEINGLM)
  add_subdirectory(ProteinGLM)
endif()

if (MODEL_QWEN3VL)
  add_subdirectory(Qwen3VL)
endif()

if (MODEL_TIMESFM)
  add_subdirectory(timesfm)
endif()

if (MODEL_MISTRAL)
  add_subdirectory(mistral)
endif()

if (MODEL_KOKORO)
  add_subdirectory(kokoro)
endif()

if (MODEL_MOLFORMER)
  add_subdirectory(molformer)
endif()

if (MODEL_COLBERTV2)
  add_subdirectory(colbertv2)
endif()

if (MODEL_MMS)
  add_subdirectory(mms)
endif()

if (MODEL_E5_MISTRAL)
  add_subdirectory(e5-mistral)
endif()

if (MODEL_CHEMBERTA)
  add_subdirectory(chemberta)
endif()

if (MODEL_SAM2)
  add_subdirectory(sam2)
endif()

if (MODEL_SMOLVLM2)
  add_subdirectory(smolvlm2)
endif()

if (MODEL_BGE_RERANKER)
  add_subdirectory(bge-reranker)
endif()

if (MODEL_PALIGEMMA)
  add_subdirectory(paligemma)
endif()
