set(TRAIN_EPOCHS 150 CACHE STRING "Checkpoint training epochs")
add_custom_command(OUTPUT ${GENERATED_DIR}/lenet-model.pth
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_DIR}
  COMMAND ${PYTHON} -m stack.models.models.lenet.prepare
          --output ${GENERATED_DIR}/lenet-model.pth --data ${GENERATED_DIR}/datasets --epochs ${TRAIN_EPOCHS}
  DEPENDS prepare.py model.py ${CMAKE_CURRENT_SOURCE_DIR}/configs/importer-param.py
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
set(CHECKPOINT ${GENERATED_DIR}/lenet-model.pth)
add_custom_command(
  OUTPUT ${GENERATED_DIR}/forward.mlir ${GENERATED_DIR}/subgraph0.mlir
         ${GENERATED_DIR}/model.json
         ${GENERATED_DIR}/lenet.payload/params.f32 ${GENERATED_DIR}/lenet.payload/weights.bin
  COMMAND ${PYTHON} -m stack.models.models.lenet.importer
          --output-dir ${GENERATED_DIR} --compiler-build ${COMPILER_BUILD}
          --design ${DESIGN_MODULE} --checkpoint ${CHECKPOINT} ${TRACE_ARGS}
  DEPENDS importer.py ${CHECKPOINT} ${DESIGN_DEPENDS}
           ${CALIBRATION_IMAGES}
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
add_custom_target(lenet-model-build DEPENDS ${GENERATED_DIR}/forward.mlir
  ${GENERATED_DIR}/subgraph0.mlir ${GENERATED_DIR}/model.json)
