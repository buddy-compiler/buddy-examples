file(GLOB_RECURSE RECIPE_FRONTEND_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/frontend/*.py")

set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")
set(WEIGHTS "${MODEL_DIR}/qwen3-8b/output/qwen3-8b")

add_custom_command(
  OUTPUT ${OUTPUT_DIR}/01-import.txt
         ${OUTPUT_DIR}/02-permuted.txt
  COMMAND ${PYTHON} buddy-pytorch-importer.py
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          ${RECIPE_FRONTEND_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${CMAKE_CURRENT_SOURCE_DIR}/frontend/permute/graph.py
          ${WEIGHTS}/config.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing the qwen3-8b graph"
  VERBATIM
)
add_custom_target(qwen3-8b-import DEPENDS ${OUTPUT_DIR}/01-import.txt ${OUTPUT_DIR}/02-permuted.txt)
