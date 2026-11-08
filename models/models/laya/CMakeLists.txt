#===------------------------------------------------------------------===#
# Download weights and prepare model.
#===------------------------------------------------------------------===#
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/laya/model.safetensors
  COMMAND ${PYTHON} buddy-pytorch-laya-prepare.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-laya-prepare.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Preparing laya weights"
  VERBATIM
)
add_custom_target(laya-prepare DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/laya/model.safetensors)

# ===----------------------------------------------------------------------===
# Export the uninstrumented graph for inspection.
# ===----------------------------------------------------------------------===
# ===----------------------------------------------------------------------===
# Export the uninstrumented graph for inspection.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt
  COMMAND ${PYTHON} buddy-pytorch-laya-graph.py
  DEPENDS buddy-pytorch-laya-graph.py
          output/laya/rl_agent_config.json
          output/laya/model.safetensors
          output/laya/encoder/config.json
          model.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Exporting laya graph"
  VERBATIM
)
add_custom_target(buddy-laya-graph DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output/buddy-graph.txt)
