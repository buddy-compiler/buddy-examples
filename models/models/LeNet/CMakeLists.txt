set(IMG_PATH ${MODEL_LENET_DIR}/images)
add_custom_command(
  OUTPUT ${MODEL_LENET_DIR}/lenet-model.pth
  COMMAND export LENET_MODEL_PATH=${MODEL_LENET_DIR} &&
          cd ${MODEL_LENET_DIR} &&
          ${Python3_EXECUTABLE} ${MODEL_LENET_DIR}/pytorch-lenet-train.py
  DEPENDS pytorch-lenet-train.py model.py
  COMMENT "Generating lenet-model.pth"
  USES_TERMINAL
)

add_custom_command(
  OUTPUT  ${MODEL_LENET_DIR}/forward.mlir ${MODEL_LENET_DIR}/subgraph0.mlir ${MODEL_LENET_DIR}/arg0.data
  COMMAND export LENET_MODEL_PATH=${MODEL_LENET_DIR} &&
          ${Python3_EXECUTABLE} ${MODEL_LENET_DIR}/buddy-lenet-import.py
  DEPENDS buddy-lenet-import.py model.py ${MODEL_LENET_DIR}/lenet-model.pth
  COMMENT "Generating forward.mlir, subgraph0.mlir and parameter files"
)

add_custom_command(
  OUTPUT ${MODEL_LENET_DIR}/subgraph0.trace.mlir
  COMMAND ${Python3_EXECUTABLE} ${MODEL_LENET_DIR}/generate_lenet_trace_mlir.py
          --input ${MODEL_LENET_DIR}/subgraph0.mlir
          --output ${MODEL_LENET_DIR}/subgraph0.trace.mlir
  DEPENDS ${MODEL_LENET_DIR}/subgraph0.mlir
          ${MODEL_LENET_DIR}/generate_lenet_trace_mlir.py
  COMMENT "Generating traced subgraph0.trace.mlir"
)

add_custom_target(lenet-model-build DEPENDS
  ${MODEL_LENET_DIR}/forward.mlir
  ${MODEL_LENET_DIR}/subgraph0.mlir
  ${MODEL_LENET_DIR}/arg0.data
  ${MODEL_LENET_DIR}/subgraph0.trace.mlir
)
