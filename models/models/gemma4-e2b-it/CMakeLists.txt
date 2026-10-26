file(GLOB MODEL_WEIGHTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/output/gemma4-e2b-it/*.safetensors"
  "${CMAKE_CURRENT_SOURCE_DIR}/output/gemma4-e2b-it/*.bin"
)

# ===----------------------------------------------------------------------===
# Prepare model weights before exporting or compiling a graph.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/gemma4-e2b-it/config.json
  COMMAND ${PYTHON} buddy-pytorch-gemma4-e2b-it-prepare.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-gemma4-e2b-it-prepare.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Preparing gemma4-e2b-it weights"
  VERBATIM
)
add_custom_target(gemma4-e2b-it-prepare DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/gemma4-e2b-it/config.json)

# ===----------------------------------------------------------------------===
# Export the uninstrumented graph for inspection.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt
  COMMAND ${PYTHON} buddy-pytorch-gemma4-e2b-it-graph.py
  DEPENDS buddy-pytorch-gemma4-e2b-it-graph.py
          ${MODEL_WEIGHTS}
          output/gemma4-e2b-it/config.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Exporting gemma4-e2b-it graph"
  VERBATIM
)
add_custom_target(buddy-gemma4-e2b-it-graph DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt)
