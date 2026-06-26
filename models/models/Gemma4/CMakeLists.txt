set(GEMMA4_MODEL_FILES
  ${MODEL_GEMMA4_DIR}/forward_prefill.mlir
  ${MODEL_GEMMA4_DIR}/subgraph0_prefill.mlir
  ${MODEL_GEMMA4_DIR}/forward_decode.mlir
  ${MODEL_GEMMA4_DIR}/subgraph0_decode.mlir
  ${MODEL_GEMMA4_DIR}/arg0.data
  ${MODEL_GEMMA4_DIR}/vocab.txt
)

foreach(model_file ${GEMMA4_MODEL_FILES})
  if(NOT EXISTS ${model_file})
    message(FATAL_ERROR "Missing Gemma4 model file: ${model_file}")
  endif()
endforeach()

add_custom_target(gemma4-model-build DEPENDS ${GEMMA4_MODEL_FILES})
