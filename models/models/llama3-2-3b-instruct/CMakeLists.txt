file(GLOB MODEL_WEIGHTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/output/llama3-2-3b-instruct/*.safetensors"
  "${CMAKE_CURRENT_SOURCE_DIR}/output/llama3-2-3b-instruct/*.bin"
)

# ===----------------------------------------------------------------------===
# Prepare model weights before exporting or compiling a graph.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/llama3-2-3b-instruct/config.json
  COMMAND ${PYTHON} buddy-pytorch-llama3-2-3b-instruct-prepare.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-llama3-2-3b-instruct-prepare.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Preparing llama3-2-3b-instruct weights"
  VERBATIM
)
add_custom_target(
  llama3-2-3b-instruct-prepare
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/llama3-2-3b-instruct/config.json
)

# ===----------------------------------------------------------------------===
# Export the uninstrumented graph for inspection.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt
  COMMAND ${PYTHON} buddy-pytorch-llama3-2-3b-instruct-graph.py
  DEPENDS buddy-pytorch-llama3-2-3b-instruct-graph.py
          ${MODEL_WEIGHTS}
          output/llama3-2-3b-instruct/config.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Exporting llama3-2-3b-instruct graph"
  VERBATIM
)
add_custom_target(buddy-llama3-2-3b-instruct-graph DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt)
