# ===----------------------------------------------------------------------===
# Prepare model weights before exporting or compiling a graph.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/resnet18/model.pth
  COMMAND ${PYTHON} buddy-pytorch-resnet18-prepare.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-resnet18-prepare.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Preparing resnet18 weights"
  VERBATIM
)
add_custom_target(resnet18-prepare DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/resnet18/model.pth)

# ===----------------------------------------------------------------------===
# Export the uninstrumented graph for inspection.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt
  COMMAND ${PYTHON} buddy-pytorch-resnet18-graph.py
  DEPENDS buddy-pytorch-resnet18-graph.py
          output/resnet18/model.pth
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Exporting resnet18 graph"
  VERBATIM
)
add_custom_target(buddy-resnet18-graph DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt)
