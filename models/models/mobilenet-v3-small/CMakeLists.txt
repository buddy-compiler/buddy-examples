set(MODEL_CONFIG_ARGS)
if(DEFINED WEIGHTS)
  list(APPEND MODEL_CONFIG_ARGS --weights "${WEIGHTS}")
endif()
set(MODEL_CONFIG_FILES ${CMAKE_CURRENT_SOURCE_DIR}/configs/model.toml ${CMAKE_CURRENT_SOURCE_DIR}/configs/importer-param.py)
add_custom_command(
  OUTPUT ${GENERATED_DIR}/forward.mlir ${GENERATED_DIR}/subgraph0.mlir
         ${GENERATED_DIR}/model.json
         ${GENERATED_DIR}/mobilenetv3.payload/params.f32 ${GENERATED_DIR}/mobilenetv3.payload/weights.bin
  COMMAND ${PYTHON} -m stack.models.models.mobilenet-v3-small.importer ${MODEL_CONFIG_ARGS}
          --output-dir ${GENERATED_DIR} --compiler-build ${COMPILER_BUILD}
          --design ${DESIGN_MODULE}  ${TRACE_ARGS}
  DEPENDS ${MODEL_CONFIG_FILES} importer.py ${CHECKPOINT} ${DESIGN_DEPENDS}
           ${CALIBRATION_IMAGES}
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
add_custom_target(mobilenetv3-model-build DEPENDS ${GENERATED_DIR}/forward.mlir
  ${GENERATED_DIR}/subgraph0.mlir ${GENERATED_DIR}/model.json)
