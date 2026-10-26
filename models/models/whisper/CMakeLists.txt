file(GLOB MODEL_WEIGHTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/output/whisper/*.safetensors"
  "${CMAKE_CURRENT_SOURCE_DIR}/output/whisper/*.bin"
)

#===------------------------------------------------------------------===#
# Download weights and prepare model.
#===------------------------------------------------------------------===#
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/whisper/config.json
  COMMAND ${PYTHON} buddy-pytorch-whisper-prepare.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-whisper-prepare.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Preparing whisper weights"
  VERBATIM
)
add_custom_target(whisper-prepare DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/whisper/config.json)

# ===----------------------------------------------------------------------===
# Export the uninstrumented graph for inspection.
# ===----------------------------------------------------------------------===
# ===----------------------------------------------------------------------===
# Export the uninstrumented graph for inspection.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt
  COMMAND ${PYTHON} buddy-pytorch-whisper-graph.py
  DEPENDS buddy-pytorch-whisper-graph.py
          ${MODEL_WEIGHTS}
          output/whisper/config.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Exporting whisper graph"
  VERBATIM
)

add_custom_target(buddy-whisper-graph DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt)
