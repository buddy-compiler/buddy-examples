set(SEQUENCE_LENGTH 32 CACHE STRING "BERT sequence length")
add_custom_command(
  OUTPUT ${GENERATED_DIR}/forward.mlir
         ${BERT_SUBGRAPH_MLIR}
         ${GENERATED_DIR}/params.f32
         ${GENERATED_DIR}/weights.bin
         ${GENERATED_DIR}/scales.bin
         ${GENERATED_DIR}/bert.rax
         ${GENERATED_DIR}/bert-parameters.h
  COMMAND ${PYTHON} -m stack.models.models.bert-base-uncased-emotion.importer
          --output-dir ${GENERATED_DIR}
          --compiler-build ${COMPILER_BUILD}
          --design ${DESIGN_MODULE}
          --partition-count ${PARTITIONS} --sequence-length ${SEQUENCE_LENGTH} ${TRACE_ARGS}
  DEPENDS ${DESIGN_DEPENDS} ${MODEL_SOURCE}/importer.py ${MODEL_SOURCE}/configs/importer-param.py
  WORKING_DIRECTORY ${REPO_ROOT}
  COMMENT "Generating BERT subgraphs and parameter files"
)
add_custom_target(bert-model-build DEPENDS
  ${GENERATED_DIR}/forward.mlir
  ${BERT_SUBGRAPH_MLIR}
  ${GENERATED_DIR}/params.f32
  ${GENERATED_DIR}/weights.bin
)
