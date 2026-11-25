file(GLOB MODEL_WEIGHTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/output/pi0-base/*.safetensors"
  "${CMAKE_CURRENT_SOURCE_DIR}/output/pi0-base/*.bin"
)

# ===----------------------------------------------------------------------===
# Prepare model weights before exporting or compiling a graph.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/pi0-base/config.json
  COMMAND ${PYTHON} buddy-pytorch-pi0-base-prepare.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-pi0-base-prepare.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Preparing pi0-base weights"
  VERBATIM
)
add_custom_target(pi0-base-prepare DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/pi0-base/config.json)
