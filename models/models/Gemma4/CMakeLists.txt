add_custom_command(
  OUTPUT ${MODEL_GEMMA4_DIR}/forward_prefill.mlir
         ${MODEL_GEMMA4_DIR}/subgraph0_prefill.mlir
         ${MODEL_GEMMA4_DIR}/forward_decode.mlir
         ${MODEL_GEMMA4_DIR}/subgraph0_decode.mlir
         ${MODEL_GEMMA4_DIR}/arg0.data
         ${MODEL_GEMMA4_DIR}/vocab.txt
  COMMAND ${Python3_EXECUTABLE} ${MODEL_GEMMA4_DIR}/import-gemma4.py
          --output-dir ${MODEL_GEMMA4_DIR}
          --precision f32
          --trace
  COMMENT "Generating Gemma4 model files..."
)

add_custom_target(gemma4-model-build DEPENDS
  ${MODEL_GEMMA4_DIR}/forward_prefill.mlir
  ${MODEL_GEMMA4_DIR}/subgraph0_prefill.mlir
  ${MODEL_GEMMA4_DIR}/forward_decode.mlir
  ${MODEL_GEMMA4_DIR}/subgraph0_decode.mlir
  ${MODEL_GEMMA4_DIR}/arg0.data
  ${MODEL_GEMMA4_DIR}/vocab.txt
)
