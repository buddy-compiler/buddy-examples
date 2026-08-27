set(IMG_PATH ${MODEL_LENET_DIR}/images)
set(MODEL_LENET_OUTPUT_DIR ${MODEL_LENET_DIR}/output)


add_custom_command(
  OUTPUT ${MODEL_LENET_OUTPUT_DIR}/lenet-model.pth
  COMMAND ${CMAKE_COMMAND} -E make_directory ${MODEL_LENET_OUTPUT_DIR}
  COMMAND export LENET_MODEL_PATH=${MODEL_LENET_OUTPUT_DIR} &&
          cd ${MODEL_LENET_OUTPUT_DIR} &&
          ${Python3_EXECUTABLE} ${MODEL_LENET_DIR}/pytorch-lenet-train.py
  DEPENDS pytorch-lenet-train.py model.py
  COMMENT "Generating lenet-model.pth"
  USES_TERMINAL
)


add_custom_command(
  OUTPUT
    ${MODEL_LENET_OUTPUT_DIR}/forward.mlir
    ${MODEL_LENET_OUTPUT_DIR}/subgraph0.mlir
    ${MODEL_LENET_OUTPUT_DIR}/lenet.rax
    ${MODEL_LENET_OUTPUT_DIR}/lenet.payload/params.f32
    ${MODEL_LENET_OUTPUT_DIR}/lenet.payload/weights.i8
    ${MODEL_LENET_OUTPUT_DIR}/lenet.payload/scales.bin
    ${MODEL_LENET_OUTPUT_DIR}/buddy-graph.txt
  COMMAND ${CMAKE_COMMAND} -E env
          LENET_MODEL_PATH=${MODEL_LENET_OUTPUT_DIR}
          BUDDY_MLIR_BUILD_DIR=$ENV{BUDDY_MLIR_BUILD_DIR}
          PYTHONPATH=${BUDDY_BINARY_DIR}/../python_packages
          ${Python3_EXECUTABLE} ${MODEL_LENET_DIR}/buddy-lenet-import.py
  DEPENDS buddy-lenet-import.py model.py ${MODEL_LENET_OUTPUT_DIR}/lenet-model.pth
  COMMENT "Generating forward.mlir, subgraph0.mlir and parameter files"
)

add_custom_target(lenet-model-build DEPENDS
  ${MODEL_LENET_OUTPUT_DIR}/forward.mlir
  ${MODEL_LENET_OUTPUT_DIR}/subgraph0.mlir
  ${MODEL_LENET_OUTPUT_DIR}/lenet.rax
  ${MODEL_LENET_OUTPUT_DIR}/lenet.payload/params.f32
  ${MODEL_LENET_OUTPUT_DIR}/lenet.payload/weights.i8
  ${MODEL_LENET_OUTPUT_DIR}/lenet.payload/scales.bin
  ${MODEL_LENET_OUTPUT_DIR}/buddy-graph.txt
)
