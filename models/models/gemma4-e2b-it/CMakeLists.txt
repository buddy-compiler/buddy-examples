set(MODEL_CONFIG_ARGS)
if(DEFINED CHECKPOINT)
  list(APPEND MODEL_CONFIG_ARGS --checkpoint "${CHECKPOINT}")
endif()
set(MODEL_CONFIG_FILES ${CMAKE_CURRENT_SOURCE_DIR}/configs/model.toml ${CMAKE_CURRENT_SOURCE_DIR}/configs/importer-param.py)
set(MODEL_OUTPUTS)
foreach(PHASE prefill decode)
  list(APPEND MODEL_OUTPUTS ${GENERATED_DIR}/forward_${PHASE}.mlir
                            ${GENERATED_DIR}/subgraph0_${PHASE}.mlir)
endforeach()
add_custom_command(
  OUTPUT ${MODEL_OUTPUTS} ${GENERATED_DIR}/arg0.data ${GENERATED_DIR}/vocab.txt
  COMMAND ${PYTHON} -m stack.models.models.gemma4-e2b-it.import-gemma4 ${MODEL_CONFIG_ARGS}
          --output-dir ${GENERATED_DIR} --compiler-build ${COMPILER_BUILD}
          ${TRACE_ARGS}
  DEPENDS ${MODEL_CONFIG_FILES} ${MODEL_SOURCE}/import-gemma4.py ${MODEL_SOURCE}/patch_decode_mlir.py
          ${DESIGN_DEPENDS}
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
add_custom_target(gemma4-model-build DEPENDS ${MODEL_OUTPUTS}
  ${GENERATED_DIR}/arg0.data ${GENERATED_DIR}/vocab.txt)
