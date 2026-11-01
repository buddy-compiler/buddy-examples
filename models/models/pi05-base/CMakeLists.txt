file(GLOB MODEL_WEIGHTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/output/pi05-base/*.safetensors"
  "${CMAKE_CURRENT_SOURCE_DIR}/output/pi05-base/*.bin"
)

# ===----------------------------------------------------------------------===
# Prepare model weights before exporting or compiling a graph.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/pi05-base/config.json
  COMMAND ${PYTHON} buddy-pytorch-pi05-base-prepare.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-pi05-base-prepare.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Preparing pi05-base weights"
  VERBATIM
)
add_custom_target(pi05-base-prepare DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/pi05-base/config.json)

# ===----------------------------------------------------------------------===
# Export the uninstrumented graph for inspection.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt
  COMMAND ${PYTHON} buddy-pytorch-pi05-base-graph.py
  DEPENDS buddy-pytorch-pi05-base-graph.py
          ${MODEL_WEIGHTS}
          output/pi05-base/config.json
          ${CMAKE_CURRENT_SOURCE_DIR}/model.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Exporting pi05-base graph"
  VERBATIM
)
add_custom_target(buddy-pi05-base-graph DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt)
