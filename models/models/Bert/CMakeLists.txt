set(BERT_CORE_COUNT 4 CACHE STRING
    "Number of homogeneous Goban Cores used to compile BERT")
if(BERT_CORE_COUNT LESS 1)
  message(FATAL_ERROR "BERT_CORE_COUNT must be positive")
endif()
math(EXPR BERT_LAST_CORE "${BERT_CORE_COUNT} - 1")
set(BERT_SUBGRAPH_MLIR)
foreach(BERT_CORE RANGE 0 ${BERT_LAST_CORE})
  list(APPEND BERT_SUBGRAPH_MLIR
       ${MODEL_BERT_DIR}/subgraph${BERT_CORE}.mlir)
endforeach()

add_custom_command(
  OUTPUT ${MODEL_BERT_DIR}/forward.mlir
         ${BERT_SUBGRAPH_MLIR}
         ${MODEL_BERT_DIR}/arg0.data
         ${MODEL_BERT_DIR}/arg1.data
  COMMAND ${Python3_EXECUTABLE} ${MODEL_BERT_DIR}/import-bert.py
          --output-dir ${MODEL_BERT_DIR}
          --core-count ${BERT_CORE_COUNT}
          --trace
  COMMENT "Generating BERT Core subgraphs and parameter files"
)

add_custom_target(bert-model-build DEPENDS
  ${MODEL_BERT_DIR}/forward.mlir
  ${BERT_SUBGRAPH_MLIR}
  ${MODEL_BERT_DIR}/arg0.data
  ${MODEL_BERT_DIR}/arg1.data
)
