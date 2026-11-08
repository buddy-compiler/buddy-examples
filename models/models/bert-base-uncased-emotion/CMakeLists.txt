file(GLOB MODEL_WEIGHTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/output/bert-base-uncased-emotion/*.safetensors"
  "${CMAKE_CURRENT_SOURCE_DIR}/output/bert-base-uncased-emotion/*.bin"
)

# ===----------------------------------------------------------------------===
# Prepare model weights before exporting or compiling a graph.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/bert-base-uncased-emotion/config.json
  COMMAND ${PYTHON} buddy-pytorch-bert-base-uncased-emotion-prepare.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-bert-base-uncased-emotion-prepare.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Preparing bert-base-uncased-emotion weights"
  VERBATIM
)
add_custom_target(
  bert-base-uncased-emotion-prepare
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/bert-base-uncased-emotion/config.json
)

# ===----------------------------------------------------------------------===
# Export the uninstrumented graph for inspection.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt
  COMMAND ${PYTHON} buddy-pytorch-bert-base-uncased-emotion-graph.py
  DEPENDS buddy-pytorch-bert-base-uncased-emotion-graph.py
          ${MODEL_WEIGHTS}
          output/bert-base-uncased-emotion/config.json
          model.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Exporting bert-base-uncased-emotion graph"
  VERBATIM
)
add_custom_target(buddy-bert-base-uncased-emotion-graph DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt)
