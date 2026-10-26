file(GLOB MODEL_WEIGHTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/output/deepseek-r1-0528-qwen3-8b/*.safetensors"
  "${CMAKE_CURRENT_SOURCE_DIR}/output/deepseek-r1-0528-qwen3-8b/*.bin"
)

# ===----------------------------------------------------------------------===
# Prepare model weights before exporting or compiling a graph.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/deepseek-r1-0528-qwen3-8b/config.json
  COMMAND ${PYTHON} buddy-pytorch-deepseek-r1-0528-qwen3-8b-prepare.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-deepseek-r1-0528-qwen3-8b-prepare.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Preparing deepseek-r1-0528-qwen3-8b weights"
  VERBATIM
)
add_custom_target(
  deepseek-r1-0528-qwen3-8b-prepare
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/deepseek-r1-0528-qwen3-8b/config.json
)

# ===----------------------------------------------------------------------===
# Export the uninstrumented graph for inspection.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt
  COMMAND ${PYTHON} buddy-pytorch-deepseek-r1-0528-qwen3-8b-graph.py
  DEPENDS buddy-pytorch-deepseek-r1-0528-qwen3-8b-graph.py
          ${MODEL_WEIGHTS}
          output/deepseek-r1-0528-qwen3-8b/config.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Exporting deepseek-r1-0528-qwen3-8b graph"
  VERBATIM
)
add_custom_target(buddy-deepseek-r1-0528-qwen3-8b-graph DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt)
