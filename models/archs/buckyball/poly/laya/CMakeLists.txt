file(GLOB_RECURSE RECIPE_FRONTEND_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/frontend/*.py")

set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")
set(WEIGHTS "${MODEL_DIR}/laya/output/laya")

add_custom_command(
  OUTPUT ${OUTPUT_DIR}/01-import.txt
         ${OUTPUT_DIR}/02-permuted.txt
  COMMAND ${PYTHON} buddy-pytorch-importer.py
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          ${RECIPE_FRONTEND_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${CMAKE_CURRENT_SOURCE_DIR}/frontend/permute/graph.py
          ${MODEL_DIR}/laya/model.py
          ${WEIGHTS}/model.safetensors
          ${WEIGHTS}/rl_agent_config.json
          ${WEIGHTS}/encoder/config.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing the laya graph"
  VERBATIM
)
add_custom_target(laya-import DEPENDS ${OUTPUT_DIR}/01-import.txt ${OUTPUT_DIR}/02-permuted.txt)
