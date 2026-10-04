set(MODEL_CONFIG_ARGS)
if(DEFINED CHECKPOINT)
  list(APPEND MODEL_CONFIG_ARGS --checkpoint "${CHECKPOINT}")
endif()
set(MODEL_CONFIG_FILES ${CMAKE_CURRENT_SOURCE_DIR}/configs/model.toml ${CMAKE_CURRENT_SOURCE_DIR}/configs/importer-param.py)
add_custom_command(
  OUTPUT ${MODEL_DEEPSEEKR1_DIR}/forward.mlir
         ${MODEL_DEEPSEEKR1_DIR}/subgraph0.mlir
         ${MODEL_DEEPSEEKR1_DIR}/arg0.data
  COMMAND ${Python3_EXECUTABLE} ${MODEL_DEEPSEEKR1_DIR}/import-deepseek-r1.py ${MODEL_CONFIG_ARGS}
          --output-dir ${MODEL_DEEPSEEKR1_DIR}
  DEPENDS ${MODEL_CONFIG_FILES}
  COMMENT "Generating forward.mlir, subgraph0.mlir and arg0.data..."
)

add_custom_target(deepseekr1-model-build DEPENDS
  ${MODEL_DEEPSEEKR1_DIR}/forward.mlir
  ${MODEL_DEEPSEEKR1_DIR}/subgraph0.mlir
  ${MODEL_DEEPSEEKR1_DIR}/arg0.data
)
