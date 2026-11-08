# ===----------------------------------------------------------------------===
# Train LeNet and save its checkpoint.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/lenet/model.pth
  COMMAND ${PYTHON} buddy-pytorch-lenet-prepare.py
            --epochs 150
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-lenet-prepare.py
          ${CMAKE_CURRENT_SOURCE_DIR}/model.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Training LeNet checkpoint"
  VERBATIM
)
add_custom_target(lenet-prepare DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/lenet/model.pth)

# ===----------------------------------------------------------------------===
# Export the uninstrumented graph for inspection.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt
  COMMAND ${PYTHON} buddy-pytorch-lenet-graph.py
  DEPENDS buddy-pytorch-lenet-graph.py
          model.py
          output/lenet/model.pth
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Exporting LeNet graph"
  VERBATIM
)
add_custom_target(buddy-lenet-graph DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt)
