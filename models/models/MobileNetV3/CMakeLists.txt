add_custom_command(
  OUTPUT ${MODEL_MOBILENETV3_DIR}/arg0.data
         ${MODEL_MOBILENETV3_DIR}/forward.mlir
         ${MODEL_MOBILENETV3_DIR}/subgraph0.mlir
  COMMAND ${Python3_EXECUTABLE} ${MODEL_MOBILENETV3_DIR}/buddy-mobilenetv3-import.py
          --output-dir ${MODEL_MOBILENETV3_DIR}
          --trace
  COMMENT "Generating forward.mlir, subgraph0.mlir and parameter files"
)

add_custom_target(mobilenetv3-model-build DEPENDS
  ${MODEL_MOBILENETV3_DIR}/forward.mlir
  ${MODEL_MOBILENETV3_DIR}/subgraph0.mlir
  ${MODEL_MOBILENETV3_DIR}/arg0.data
)
