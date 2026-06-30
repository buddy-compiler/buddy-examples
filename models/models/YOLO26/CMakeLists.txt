add_custom_command(
  OUTPUT ${MODEL_YOLO26_DIR}/arg0.data
         ${MODEL_YOLO26_DIR}/forward.mlir
         ${MODEL_YOLO26_DIR}/subgraph0.mlir
  COMMAND ${Python3_EXECUTABLE} ${MODEL_YOLO26_DIR}/import-yolo26n.py
          --output-dir ${MODEL_YOLO26_DIR}
          --trace
  COMMENT "Generating YOLO26 model files..."
)

add_custom_target(yolo26-model-build DEPENDS
  ${MODEL_YOLO26_DIR}/forward.mlir
  ${MODEL_YOLO26_DIR}/subgraph0.mlir
  ${MODEL_YOLO26_DIR}/arg0.data
)
