add_custom_command(
  OUTPUT ${MODEL_ALEXNET_DIR}/arg0.data
         ${MODEL_ALEXNET_DIR}/forward.mlir
         ${MODEL_ALEXNET_DIR}/subgraph0.mlir
  COMMAND ${Python3_EXECUTABLE} ${MODEL_ALEXNET_DIR}/buddy-alexnet-import.py
          --output-dir ${MODEL_ALEXNET_DIR}
  COMMENT "Generating forward.mlir, subgraph0.mlir and parameter files"
)

add_custom_target(alexnet-model-build DEPENDS
  ${MODEL_ALEXNET_DIR}/forward.mlir
  ${MODEL_ALEXNET_DIR}/subgraph0.mlir
  ${MODEL_ALEXNET_DIR}/arg0.data
)
