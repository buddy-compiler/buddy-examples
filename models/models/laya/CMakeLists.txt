set(SEQUENCE_LENGTH 128 CACHE STRING "Laya input token capacity")
set(OPTIONS 16 CACHE STRING "Laya candidate capacity")
set(STAGES embedding)
foreach(LAYER RANGE 0 27)
  list(APPEND STAGES attention_${LAYER} ffn_${LAYER})
endforeach()
list(APPEND STAGES typed head_attention_0 head_ffn_0 head_attention_1 head_ffn_1 scorer action)
set(GRAPHS)
foreach(STAGE IN LISTS STAGES)
  list(APPEND GRAPHS ${GENERATED_DIR}/${STAGE}.mlir ${GENERATED_DIR}/${STAGE}-forward.mlir)
endforeach()
add_custom_command(
  OUTPUT ${GRAPHS} ${GENERATED_DIR}/stages.h ${GENERATED_DIR}/model.json
  COMMAND ${PYTHON} -m stack.models.models.laya.importer
          --output-dir ${GENERATED_DIR} --compiler-build ${COMPILER_BUILD}
          --design ${DESIGN_MODULE} --sequence-length ${SEQUENCE_LENGTH} --options ${OPTIONS}
  DEPENDS importer.py model.py inputs.py ${DESIGN_FILES}
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
add_custom_target(laya-model-build DEPENDS ${GRAPHS} ${GENERATED_DIR}/stages.h ${GENERATED_DIR}/model.json)
set(STAGES ${STAGES} PARENT_SCOPE)
