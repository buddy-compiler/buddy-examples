file(GLOB MODEL_WEIGHTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/output/qwen3-0-6b/*.safetensors"
  "${CMAKE_CURRENT_SOURCE_DIR}/output/qwen3-0-6b/*.bin"
)

# ===----------------------------------------------------------------------===
# Prepare model weights before exporting or compiling a graph.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/qwen3-0-6b/config.json
  COMMAND ${PYTHON} buddy-pytorch-qwen3-0-6b-prepare.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-qwen3-0-6b-prepare.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Preparing qwen3-0-6b weights"
  VERBATIM
)
add_custom_target(qwen3-0-6b-prepare DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/qwen3-0-6b/config.json)

# ===----------------------------------------------------------------------===
# Export the uninstrumented graph for inspection.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt
  COMMAND ${PYTHON} buddy-pytorch-qwen3-0-6b-graph.py
  DEPENDS buddy-pytorch-qwen3-0-6b-graph.py
          ${MODEL_WEIGHTS}
          output/qwen3-0-6b/config.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Exporting qwen3-0-6b graph"
  VERBATIM
)
add_custom_target(buddy-qwen3-0-6b-graph DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt)
