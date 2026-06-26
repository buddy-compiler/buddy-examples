add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
  COMMAND ${CMAKE_COMMAND} -E env
          TRANSFORMERS_OFFLINE=1
          HF_HUB_OFFLINE=1
          ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-whisper.py
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}
          --trace
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/trace/trace.toml
  COMMENT "Generating forward.mlir, subgraph0.mlir and arg0.data..."
)

add_custom_target(whisper-model-build DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
)
