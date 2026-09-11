set(MODEL_YOLO26_OUTPUT_DIR ${MODEL_YOLO26_DIR}/output)
set(YOLO26_CALIBRATION_IMAGES
  ${MODEL_YOLO26_DIR}/../ResNet18/images/dog.png
  ${MODEL_YOLO26_DIR}/../ResNet18/images/ice-cream.png
  ${MODEL_YOLO26_DIR}/../ResNet18/images/traffic-light.png
  ${MODEL_YOLO26_DIR}/../ResNet18/images/kite.png
  ${MODEL_YOLO26_DIR}/../ResNet18/images/curtain.png)

add_custom_command(
  OUTPUT ${MODEL_YOLO26_OUTPUT_DIR}/yolo26.rax
         ${MODEL_YOLO26_OUTPUT_DIR}/yolo26.payload/params.f32
         ${MODEL_YOLO26_OUTPUT_DIR}/yolo26.payload/weights.i8
         ${MODEL_YOLO26_OUTPUT_DIR}/yolo26.payload/scales.bin
         ${MODEL_YOLO26_OUTPUT_DIR}/yolo26.payload/quant-index.json
         ${MODEL_YOLO26_OUTPUT_DIR}/forward.mlir
         ${MODEL_YOLO26_OUTPUT_DIR}/subgraph0.mlir
         ${MODEL_YOLO26_OUTPUT_DIR}/calibration-manifest.json
  COMMAND ${CMAKE_COMMAND} -E make_directory ${MODEL_YOLO26_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E env
          BUDDY_MLIR_BUILD_DIR=$ENV{BUDDY_MLIR_BUILD_DIR}
          YOLO26N_MODEL_PATH=${MODEL_YOLO26_OUTPUT_DIR}/yolo26n.pt
          PYTHONPATH=${BUDDY_BINARY_DIR}/../python_packages
          LD_LIBRARY_PATH=${BUDDY_BINARY_DIR}/../python_packages/buddy_mlir/_mlir_libs:${BUDDY_BINARY_DIR}/../lib
          ${Python3_EXECUTABLE} ${MODEL_YOLO26_DIR}/import-yolo26n.py
          --output-dir ${MODEL_YOLO26_OUTPUT_DIR}
          --img-size 640
          --calibration-images ${YOLO26_CALIBRATION_IMAGES}
          # --trace
  DEPENDS ${MODEL_YOLO26_DIR}/import-yolo26n.py
          ${MODEL_YOLO26_DIR}/calibration.py
          ${YOLO26_CALIBRATION_IMAGES}
          ${MODEL_YOLO26_OUTPUT_DIR}/yolo26n.pt
          ${MODEL_YOLO26_DIR}/../../../framework/quant/core/activation.py
          ${MODEL_YOLO26_DIR}/../../../framework/quant/core/importer.py
          ${BUDDY_BINARY_DIR}/../python_packages/buddy/compiler/ops/linalg.py
          ${BUDDY_BINARY_DIR}/../python_packages/buddy/compiler/ops/tosa.py
  COMMENT "Generating YOLO26 model files..."
  VERBATIM
)

add_custom_target(yolo26-model-build DEPENDS
  ${MODEL_YOLO26_OUTPUT_DIR}/forward.mlir
  ${MODEL_YOLO26_OUTPUT_DIR}/subgraph0.mlir
  ${MODEL_YOLO26_OUTPUT_DIR}/yolo26.rax
  ${MODEL_YOLO26_OUTPUT_DIR}/yolo26.payload/params.f32
  ${MODEL_YOLO26_OUTPUT_DIR}/yolo26.payload/weights.i8
  ${MODEL_YOLO26_OUTPUT_DIR}/yolo26.payload/scales.bin
  ${MODEL_YOLO26_OUTPUT_DIR}/yolo26.payload/quant-index.json
  ${MODEL_YOLO26_OUTPUT_DIR}/calibration-manifest.json
)
