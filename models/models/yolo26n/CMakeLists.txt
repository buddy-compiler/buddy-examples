add_custom_command(OUTPUT ${GENERATED_DIR}/yolo26n.pt
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_DIR}
  COMMAND ${CMAKE_COMMAND} -E chdir ${GENERATED_DIR} ${PYTHON} -c
          "from ultralytics import YOLO; YOLO('yolo26n.pt')"
  VERBATIM)
set(CHECKPOINT ${GENERATED_DIR}/yolo26n.pt)
add_custom_command(
  OUTPUT ${GENERATED_DIR}/forward.mlir ${GENERATED_DIR}/subgraph0.mlir
         ${GENERATED_DIR}/model.json
         ${GENERATED_DIR}/yolo26.payload/params.f32 ${GENERATED_DIR}/yolo26.payload/weights.bin
  COMMAND ${PYTHON} -m stack.models.models.yolo26n.importer
          --output-dir ${GENERATED_DIR} --compiler-build ${COMPILER_BUILD}
          --design ${DESIGN_MODULE} --checkpoint ${CHECKPOINT} --img-size 640 --calibration-images ${CALIBRATION_IMAGES} ${TRACE_ARGS}
  DEPENDS importer.py ${CHECKPOINT} ${DESIGN_DEPENDS}
           ${CALIBRATION_IMAGES}
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
add_custom_target(yolo26-model-build DEPENDS ${GENERATED_DIR}/forward.mlir
  ${GENERATED_DIR}/subgraph0.mlir ${GENERATED_DIR}/model.json)
