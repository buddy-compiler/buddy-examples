file(GLOB_RECURSE RECIPE_FRONTEND_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/frontend/*.py")

set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")
set(WEIGHTS "${MODEL_DIR}/laya/output/laya/typed-decisions")

# ===----------------------------------------------------------------------===
# Calibrate the Linear input and attention operand scales on typed-decision
# questions; rejected when the decision accuracy drops more than 1 point.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/calibration.json
  COMMAND ${PYTHON} buddy-pytorch-calibrate.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-calibrate.py
          ${STACK_ROOT}/compiler/frontend/quant.py
          ${MODEL_DIR}/laya/model.py
          ${MODEL_DIR}/laya/data.py
          ${WEIGHTS}/model.safetensors
          ${WEIGHTS}/rl_agent_config.json
          ${WEIGHTS}/encoder/config.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Calibrating laya activations"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Import the laya graph, permute and quantize every matrix product to INT8.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/01-import.txt
         ${OUTPUT_DIR}/02-permuted.txt
         ${OUTPUT_DIR}/03-quanted.txt
         ${OUTPUT_DIR}/arg0.data
         ${OUTPUT_DIR}/arg1.data
         ${OUTPUT_DIR}/arg2.data
  COMMAND ${PYTHON} buddy-pytorch-importer.py
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          ${RECIPE_FRONTEND_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${CMAKE_CURRENT_SOURCE_DIR}/frontend/permute/graph.py
          ${MODEL_DIR}/laya/model.py
          ${WEIGHTS}/model.safetensors
          ${WEIGHTS}/rl_agent_config.json
          ${WEIGHTS}/encoder/config.json
          ${OUTPUT_DIR}/calibration.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing the laya graph"
  VERBATIM
)
add_custom_target(
  laya-import
  DEPENDS ${OUTPUT_DIR}/01-import.txt ${OUTPUT_DIR}/02-permuted.txt ${OUTPUT_DIR}/03-quanted.txt
)
