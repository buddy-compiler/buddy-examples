# Whisper model import for accelerator backends (forward.mlir / subgraph0.mlir).
# Host .rax / buddy_add_model lives in buddy-mlir; do not use CMAKE_SOURCE_DIR
# tools/buddy-codegen here — bb-tests nests ModelTest under a different source root.

set(BUDDY_WHISPER_SPEC
  "${CMAKE_CURRENT_SOURCE_DIR}/specs/base.json"
  CACHE FILEPATH "Whisper variant spec JSON, e.g. specs/base.json")

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
  COMMAND ${CMAKE_COMMAND} -E env
          TRANSFORMERS_OFFLINE=1
          HF_HUB_OFFLINE=1
          ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/codegen/import-whisper.py
          --spec ${BUDDY_WHISPER_SPEC}
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}
          --no-layer-partitioned
  COMMENT "Generating Whisper forward.mlir, subgraph0.mlir and arg0.data..."
  VERBATIM
)

add_custom_target(whisper-model-build DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
)
