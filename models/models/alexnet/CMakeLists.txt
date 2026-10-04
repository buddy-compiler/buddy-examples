set(MODEL_CONFIG_ARGS)
if(DEFINED WEIGHTS)
  list(APPEND MODEL_CONFIG_ARGS --weights "${WEIGHTS}")
endif()
set(MODEL_CONFIG_FILES ${CMAKE_CURRENT_SOURCE_DIR}/configs/model.toml ${CMAKE_CURRENT_SOURCE_DIR}/configs/importer-param.py)
set(KERNEL_MLIR)
foreach(GRAPH IN LISTS KERNEL_GRAPHS)
  list(APPEND KERNEL_MLIR ${GENERATED_DIR}/${GRAPH}.mlir)
endforeach()
add_custom_command(
  OUTPUT ${GENERATED_DIR}/forward.mlir ${KERNEL_MLIR}
         ${GENERATED_DIR}/model.json ${GENERATED_DIR}/input.f32 ${GENERATED_DIR}/reference.f32
         ${GENERATED_DIR}/alexnet.payload/params.f32 ${GENERATED_DIR}/alexnet.payload/weights.bin
  COMMAND ${PYTHON} -m stack.models.models.alexnet.importer ${MODEL_CONFIG_ARGS}
          --output-dir ${GENERATED_DIR} --compiler-build ${COMPILER_BUILD}
          --design ${DESIGN_MODULE} --chip ${CHIP} ${TRACE_ARGS}
  DEPENDS ${MODEL_CONFIG_FILES} importer.py prepare.py images/dog.bmp ${DESIGN_DEPENDS}
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
add_custom_target(alexnet-model-build DEPENDS ${GENERATED_DIR}/forward.mlir
  ${KERNEL_MLIR} ${GENERATED_DIR}/model.json)
