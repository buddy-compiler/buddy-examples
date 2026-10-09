set(MODEL_CONFIG_ARGS)
if(DEFINED CHECKPOINT)
  list(APPEND MODEL_CONFIG_ARGS --checkpoint "${CHECKPOINT}")
endif()
if(DEFINED REVISION)
  list(APPEND MODEL_CONFIG_ARGS --revision "${REVISION}")
endif()
set(MODEL_CONFIG_FILES ${CMAKE_CURRENT_SOURCE_DIR}/configs/model.toml ${CMAKE_CURRENT_SOURCE_DIR}/configs/importer-param.py)
set(KERNEL_MLIR)
foreach(GRAPH IN LISTS KERNEL_GRAPHS)
  list(APPEND KERNEL_MLIR ${GENERATED_DIR}/${GRAPH}.mlir)
endforeach()
add_custom_command(
  OUTPUT ${KERNEL_MLIR} ${GENERATED_DIR}/mixer-parameters.h
         ${GENERATED_DIR}/model.json ${GENERATED_DIR}/input.f32 ${GENERATED_DIR}/reference.f32
         ${GENERATED_DIR}/mixer.payload/params.f32 ${GENERATED_DIR}/mixer.payload/weights.bin
  COMMAND ${PYTHON} -m stack.models.models.mlp-mixer.importer ${MODEL_CONFIG_ARGS}
          --output-dir ${GENERATED_DIR} --compiler-build ${COMPILER_BUILD}
          --design ${DESIGN_MODULE} --chip ${CHIP} ${TRACE_ARGS}
  DEPENDS ${MODEL_CONFIG_FILES} importer.py prepare.py ../alexnet/images/dog.bmp ${DESIGN_DEPENDS}
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
add_custom_target(mixer-model-build DEPENDS ${KERNEL_MLIR} ${GENERATED_DIR}/mixer-parameters.h ${GENERATED_DIR}/model.json)
