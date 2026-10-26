#===------------------------------------------------------------------===#
# Download weights and prepare model.
#===------------------------------------------------------------------===#
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/yolo26n/yolo26n.pt
  COMMAND ${PYTHON} buddy-pytorch-yolo26n-prepare.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-yolo26n-prepare.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Preparing yolo26n weights"
  VERBATIM
)
add_custom_target(yolo26n-prepare DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/yolo26n/yolo26n.pt)

# ===----------------------------------------------------------------------===
# Export the uninstrumented graph for inspection.
# ===----------------------------------------------------------------------===
# ===----------------------------------------------------------------------===
# Export the uninstrumented graph for inspection.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt
  COMMAND ${PYTHON} buddy-pytorch-yolo26n-graph.py
  DEPENDS buddy-pytorch-yolo26n-graph.py
          output/yolo26n/yolo26n.pt
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Exporting yolo26n graph"
  VERBATIM
)

add_custom_target(buddy-yolo26n-graph DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt)
