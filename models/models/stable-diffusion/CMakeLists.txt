# ===----------------------------------------------------------------------===
# Prepare model weights before exporting or compiling a graph.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/stable-diffusion/model_index.json
  COMMAND ${PYTHON} buddy-pytorch-stable-diffusion-prepare.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-stable-diffusion-prepare.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Preparing stable-diffusion weights"
  VERBATIM
)
add_custom_target(stable-diffusion-prepare DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/stable-diffusion/model_index.json)

# ===----------------------------------------------------------------------===
# Export the uninstrumented graph for inspection.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt
  COMMAND ${PYTHON} buddy-pytorch-stable-diffusion-graph.py
  DEPENDS buddy-pytorch-stable-diffusion-graph.py
          output/stable-diffusion/model_index.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Exporting stable-diffusion graph"
  VERBATIM
)
add_custom_target(buddy-stable-diffusion-graph DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt)
