add_custom_command(
  OUTPUT ${MODEL_QWEN3_DIR}/forward_prefill_0_6b.mlir
         ${MODEL_QWEN3_DIR}/subgraph0_prefill_0_6b.mlir
         ${MODEL_QWEN3_DIR}/forward_decode_0_6b.mlir
         ${MODEL_QWEN3_DIR}/subgraph0_decode_0_6b.mlir
         ${MODEL_QWEN3_DIR}/arg0_0_6b.data
         ${MODEL_QWEN3_DIR}/arg1_0_6b.data
  COMMAND ${Python3_EXECUTABLE} ${MODEL_QWEN3_DIR}/import-qwen3.py
          --output-dir ${MODEL_QWEN3_DIR}
          --precision f32
          --max-cache-len 16
          --prefill-len 16
          --trace
  COMMENT "Generating Qwen3 model files..."
)

add_custom_target(qwen3-model-build DEPENDS
  ${MODEL_QWEN3_DIR}/forward_prefill_0_6b.mlir
  ${MODEL_QWEN3_DIR}/subgraph0_prefill_0_6b.mlir
  ${MODEL_QWEN3_DIR}/forward_decode_0_6b.mlir
  ${MODEL_QWEN3_DIR}/subgraph0_decode_0_6b.mlir
  ${MODEL_QWEN3_DIR}/arg0_0_6b.data
  ${MODEL_QWEN3_DIR}/arg1_0_6b.data
)
