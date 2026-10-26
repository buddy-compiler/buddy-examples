#===------------------------------------------------------------------===#
# Download weights and prepare model.
#===------------------------------------------------------------------===#
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/alexnet/model.pth
  COMMAND ${PYTHON} buddy-pytorch-alexnet-prepare.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-alexnet-prepare.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Preparing alexnet weights"
  VERBATIM
)
add_custom_target(alexnet-prepare DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/alexnet/model.pth)

# ===----------------------------------------------------------------------===
# Export the uninstrumented graph for inspection.
# ===----------------------------------------------------------------------===
# ===----------------------------------------------------------------------===
# Export the uninstrumented graph for inspection.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt
  COMMAND ${PYTHON} buddy-pytorch-alexnet-graph.py
  DEPENDS buddy-pytorch-alexnet-graph.py
          output/alexnet/model.pth
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Exporting alexnet graph"
  VERBATIM
)

add_custom_target(buddy-alexnet-graph DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt)
