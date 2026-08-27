set(MODEL_YOLO26_OUTPUT_DIR ${MODEL_YOLO26_DIR}/output)

add_custom_command(
  OUTPUT ${MODEL_YOLO26_OUTPUT_DIR}/yolo26.rax
         ${MODEL_YOLO26_OUTPUT_DIR}/yolo26.payload/params.f32
         ${MODEL_YOLO26_OUTPUT_DIR}/yolo26.payload/weights.i8
         ${MODEL_YOLO26_OUTPUT_DIR}/yolo26.payload/scales.bin
         ${MODEL_YOLO26_OUTPUT_DIR}/forward.mlir
         ${MODEL_YOLO26_OUTPUT_DIR}/subgraph0.mlir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${MODEL_YOLO26_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E env
          BUDDY_MLIR_BUILD_DIR=$ENV{BUDDY_MLIR_BUILD_DIR}
          PYTHONPATH=${BUDDY_BINARY_DIR}/../python_packages
          ${Python3_EXECUTABLE} ${MODEL_YOLO26_DIR}/import-yolo26n.py
          --output-dir ${MODEL_YOLO26_OUTPUT_DIR}
          --trace
  WORKING_DIRECTORY ${MODEL_YOLO26_OUTPUT_DIR}
  COMMENT "Generating YOLO26 model files..."
)

add_custom_target(yolo26-model-build DEPENDS
  ${MODEL_YOLO26_OUTPUT_DIR}/forward.mlir
  ${MODEL_YOLO26_OUTPUT_DIR}/subgraph0.mlir
  ${MODEL_YOLO26_OUTPUT_DIR}/yolo26.rax
  ${MODEL_YOLO26_OUTPUT_DIR}/yolo26.payload/params.f32
  ${MODEL_YOLO26_OUTPUT_DIR}/yolo26.payload/weights.i8
  ${MODEL_YOLO26_OUTPUT_DIR}/yolo26.payload/scales.bin
)
