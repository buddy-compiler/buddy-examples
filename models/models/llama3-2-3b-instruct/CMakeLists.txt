add_custom_command(
  OUTPUT ${LLAMA_MLIR} ${GENERATED_DIR}/model.json
         ${GENERATED_DIR}/llama-parameters.h
  BYPRODUCTS ${GENERATED_DIR}/tokenizer/config.json
             ${GENERATED_DIR}/tokenizer/tokenizer.json
             ${GENERATED_DIR}/tokenizer/tokenizer_config.json
  COMMAND ${PYTHON} -m stack.models.models.llama3-2-3b-instruct.importer
          --output-dir ${GENERATED_DIR}
          --compiler-build ${COMPILER_BUILD}
          --design ${DESIGN_MODULE}
          --prefill-len ${PREFILL_LENGTH}
          --max-cache-len ${CACHE_LENGTH}
          --parts ${PARTS}
          --model ${CHECKPOINT}
  DEPENDS ${DESIGN_DEPENDS} ${MODEL_SOURCE}/importer.py ${MODEL_SOURCE}/configs/importer-param.py ${MODEL_SOURCE}/stages.py
  WORKING_DIRECTORY ${REPO_ROOT}
  VERBATIM)
add_custom_target(llama-model-build DEPENDS ${LLAMA_MLIR}
  ${GENERATED_DIR}/model.json ${GENERATED_DIR}/llama-parameters.h)
