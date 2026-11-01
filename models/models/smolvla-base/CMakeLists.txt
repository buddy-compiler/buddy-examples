file(GLOB MODEL_WEIGHTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/output/smolvla-base/*.safetensors"
  "${CMAKE_CURRENT_SOURCE_DIR}/output/smolvla-base/*.bin"
)

# ===----------------------------------------------------------------------===
# Prepare model weights before exporting or compiling a graph.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/smolvla-base/config.json
  COMMAND ${PYTHON} buddy-pytorch-smolvla-base-prepare.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-smolvla-base-prepare.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Preparing smolvla-base weights"
  VERBATIM
)
add_custom_target(smolvla-base-prepare DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/smolvla-base/config.json)

# ===----------------------------------------------------------------------===
# Export the uninstrumented graph for inspection.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt
  COMMAND ${PYTHON} buddy-pytorch-smolvla-base-graph.py
  DEPENDS buddy-pytorch-smolvla-base-graph.py
          ${MODEL_WEIGHTS}
          output/smolvla-base/config.json
          ${CMAKE_CURRENT_SOURCE_DIR}/model.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Exporting smolvla-base graph"
  VERBATIM
)
add_custom_target(buddy-smolvla-base-graph DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt)
