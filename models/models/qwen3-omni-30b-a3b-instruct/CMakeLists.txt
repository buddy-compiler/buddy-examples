set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

add_custom_command(
  OUTPUT ${OUTPUT_DIR}/checkpoint.json
  COMMAND ${PYTHON} prepare.py
            --output-dir ${OUTPUT_DIR}
            --checkpoint "Qwen/Qwen3-Omni-30B-A3B-Instruct"
            --revision "26291f793822fb6be9555850f06dfe95f2d7e695"
  DEPENDS ${MODEL_DIR}/qwen3-omni-30b-a3b-instruct/configs/importer-param.py
          ${MODEL_DIR}/qwen3-omni-30b-a3b-instruct/prepare.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)
add_custom_target(qwen3-omni-30b-a3b-instruct-prepare DEPENDS ${OUTPUT_DIR}/checkpoint.json)
