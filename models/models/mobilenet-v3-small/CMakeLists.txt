add_custom_command(
  OUTPUT ${GENERATED_DIR}/forward.mlir ${GENERATED_DIR}/subgraph0.mlir
         ${GENERATED_DIR}/model.json
         ${GENERATED_DIR}/mobilenetv3.payload/params.f32 ${GENERATED_DIR}/mobilenetv3.payload/weights.bin
  COMMAND ${PYTHON} -m stack.models.models.mobilenet-v3-small.importer
          --output-dir ${GENERATED_DIR} --compiler-build ${COMPILER_BUILD}
          --design ${DESIGN_MODULE}  ${TRACE_ARGS}
  DEPENDS importer.py ${CHECKPOINT} ${DESIGN_DEPENDS}
           ${CALIBRATION_IMAGES}
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
add_custom_target(mobilenetv3-model-build DEPENDS ${GENERATED_DIR}/forward.mlir
  ${GENERATED_DIR}/subgraph0.mlir ${GENERATED_DIR}/model.json)
