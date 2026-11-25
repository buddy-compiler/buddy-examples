file(GLOB MODEL_WEIGHTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/output/pi0-fast-base/*.safetensors"
  "${CMAKE_CURRENT_SOURCE_DIR}/output/pi0-fast-base/*.bin"
)

# ===----------------------------------------------------------------------===
# Prepare model weights before exporting or compiling a graph.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/pi0-fast-base/config.json
  COMMAND ${PYTHON} buddy-pytorch-pi0-fast-base-prepare.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-pi0-fast-base-prepare.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Preparing pi0-fast-base weights"
  VERBATIM
)
add_custom_target(pi0-fast-base-prepare DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/pi0-fast-base/config.json)
