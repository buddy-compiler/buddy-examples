# ===----------------------------------------------------------------------===
# Prepare model weights before exporting or compiling a graph.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/minimax-h3-fl2va/config.json
  COMMAND ${PYTHON} buddy-pytorch-minimax-h3-fl2va-prepare.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-minimax-h3-fl2va-prepare.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Preparing minimax-h3-fl2va weights"
  VERBATIM
)
add_custom_target(minimax-h3-fl2va-prepare DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/minimax-h3-fl2va/config.json)

# ===----------------------------------------------------------------------===
# Export the uninstrumented graph for inspection.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt
  COMMAND ${PYTHON} buddy-pytorch-minimax-h3-fl2va-graph.py
  DEPENDS buddy-pytorch-minimax-h3-fl2va-graph.py
          output/minimax-h3-fl2va/FL2VA/model_index.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Exporting minimax-h3-fl2va graph"
  VERBATIM
)
add_custom_target(buddy-minimax-h3-fl2va-graph DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt)
