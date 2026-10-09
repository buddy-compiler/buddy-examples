add_custom_command(
  OUTPUT ${QWEN_MLIR} ${GENERATED_DIR}/model.json
         ${GENERATED_DIR}/qwen-parameters.h
  BYPRODUCTS ${GENERATED_DIR}/tokenizer/config.json
             ${GENERATED_DIR}/tokenizer/tokenizer.json
             ${GENERATED_DIR}/tokenizer/tokenizer_config.json
  COMMAND ${PYTHON} -m stack.models.models.qwen3-8b.importer
          --output-dir ${GENERATED_DIR}
          --compiler-build ${COMPILER_BUILD}
          --design ${DESIGN_MODULE}
          --prefill-len ${PREFILL_LENGTH}
          --max-cache-len ${CACHE_LENGTH}
          --parts ${PARTS}
          --model ${CHECKPOINT}
  DEPENDS ${DESIGN_DEPENDS} ${MODEL_SOURCE}/importer.py ${MODEL_SOURCE}/configs/importer-param.py ${MODEL_SOURCE}/stages.py ${MODEL_SOURCE}/ffn.py ${MODEL_SOURCE}/output.py ${MODEL_SOURCE}/attention.py ${MODEL_SOURCE}/parameters.py
  WORKING_DIRECTORY ${REPO_ROOT}
  VERBATIM)
add_custom_target(qwen-model-build DEPENDS ${QWEN_MLIR}
  ${GENERATED_DIR}/model.json ${GENERATED_DIR}/qwen-parameters.h)
