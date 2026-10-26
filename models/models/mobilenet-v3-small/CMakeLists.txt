# ===----------------------------------------------------------------------===
# Prepare model weights before exporting or compiling a graph.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/mobilenet-v3-small/model.pth
  COMMAND ${PYTHON} buddy-pytorch-mobilenet-v3-small-prepare.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-mobilenet-v3-small-prepare.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Preparing mobilenet-v3-small weights"
  VERBATIM
)
add_custom_target(mobilenet-v3-small-prepare DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/mobilenet-v3-small/model.pth)

# ===----------------------------------------------------------------------===
# Export the uninstrumented graph for inspection.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt
  COMMAND ${PYTHON} buddy-pytorch-mobilenet-v3-small-graph.py
  DEPENDS buddy-pytorch-mobilenet-v3-small-graph.py
          output/mobilenet-v3-small/model.pth
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Exporting mobilenet-v3-small graph"
  VERBATIM
)
add_custom_target(buddy-mobilenet-v3-small-graph DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt)
