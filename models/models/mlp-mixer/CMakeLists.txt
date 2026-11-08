# ===----------------------------------------------------------------------===
# Prepare model weights before exporting or compiling a graph.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/mlp-mixer/model.safetensors
  COMMAND ${PYTHON} buddy-pytorch-mlp-mixer-prepare.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-mlp-mixer-prepare.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Preparing mlp-mixer weights"
  VERBATIM
)
add_custom_target(mlp-mixer-prepare DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/mlp-mixer/model.safetensors)

# ===----------------------------------------------------------------------===
# Export the uninstrumented graph for inspection.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt
  COMMAND ${PYTHON} buddy-pytorch-mlp-mixer-graph.py
  DEPENDS buddy-pytorch-mlp-mixer-graph.py
          output/mlp-mixer/model.safetensors
          model.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Exporting mlp-mixer graph"
  VERBATIM
)
add_custom_target(buddy-mlp-mixer-graph DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt)
