set(MODEL_CONFIG_ARGS)
list(APPEND MODEL_CONFIG_ARGS --prefill-length "${PREFILL_LENGTH}")
list(APPEND MODEL_CONFIG_ARGS --run-config "${DESIGN_DIR}/configs/running-param.toml"
  --isa-dir "${ISA_DIR}" --kernel-passes "${GEMMA_KERNEL_PIPELINE}")
if(DEFINED CHECKPOINT)
  list(APPEND MODEL_CONFIG_ARGS --checkpoint "${CHECKPOINT}")
endif()
set(MODEL_CONFIG_FILES ${CMAKE_CURRENT_SOURCE_DIR}/configs/model.toml ${CMAKE_CURRENT_SOURCE_DIR}/configs/importer-param.py)
set(MODEL_OUTPUTS ${GENERATED_DIR}/tasks/manifest.json
  ${GENERATED_DIR}/tasks/task_api.h ${GENERATED_DIR}/tasks/task_index.cpp
  ${GENERATED_DIR}/tasks/CMakeLists.txt)
add_custom_command(
  OUTPUT ${MODEL_OUTPUTS} ${GENERATED_DIR}/model.json ${GENERATED_DIR}/gemma-parameters.h ${GENERATED_DIR}/vocab.txt
         ${GENERATED_DIR}/prefill/params.f32 ${GENERATED_DIR}/prefill/weights.bin
         ${GENERATED_DIR}/decode/params.f32 ${GENERATED_DIR}/decode/weights.bin
  COMMAND ${PYTHON} -m stack.models.models.gemma4-e2b-it.import-gemma4 ${MODEL_CONFIG_ARGS}
          --chip ${CHIP} --design-module ${DESIGN_MODULE} --output-dir ${GENERATED_DIR} --compiler-build ${COMPILER_BUILD}
          ${TRACE_ARGS}
  DEPENDS ${MODEL_CONFIG_FILES} ${MODEL_SOURCE}/import-gemma4.py ${MODEL_SOURCE}/abi.py ${MODEL_SOURCE}/cache.py ${MODEL_SOURCE}/native.py
          ${DESIGN_DEPENDS}
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
add_custom_target(gemma4-model-build DEPENDS ${MODEL_OUTPUTS}
  ${GENERATED_DIR}/model.json ${GENERATED_DIR}/gemma-parameters.h ${GENERATED_DIR}/vocab.txt)
