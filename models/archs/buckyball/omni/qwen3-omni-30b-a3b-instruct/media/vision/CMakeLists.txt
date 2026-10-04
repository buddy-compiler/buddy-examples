set(VISION_DIR "${GENERATED_DIR}/vision")
set(VISION_STAGES)
foreach(COUNT 4 16 64 256)
  foreach(KIND patch attention mlp merge deepstack)
    list(APPEND VISION_STAGES vision_${KIND}_${COUNT})
  endforeach()
endforeach()
set(VISION_FILES)
foreach(STAGE IN LISTS VISION_STAGES)
  list(APPEND VISION_FILES ${VISION_DIR}/${STAGE}.mlir ${VISION_DIR}/${STAGE}-forward.mlir)
endforeach()
add_custom_command(OUTPUT ${VISION_FILES} ${VISION_DIR}/kernels.json ${VISION_DIR}/vision-parameters.h
  COMMAND ${PYTHON} -m ${DESIGN_MODULE}.trace.vision
          --checkpoint-config ${GENERATED_DIR}/checkpoint.json
          --compiler-build ${COMPILER_BUILD} --output ${VISION_DIR}
  DEPENDS ${GENERATED_DIR}/checkpoint.json ${MODEL_SOURCE}/vision.py ${MODEL_SOURCE}/weights.py
          ${CMAKE_CURRENT_SOURCE_DIR}/trace/vision.py ${GRAPH_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/configs/compiler-param.toml
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
add_custom_command(OUTPUT ${GENERATED_DIR}/weights/vision-placement.json
  COMMAND ${PYTHON} -m ${DESIGN_MODULE}.quant.vision
          --checkpoint-config ${GENERATED_DIR}/checkpoint.json --compiler-build ${COMPILER_BUILD}
          --kernels ${VISION_DIR}/kernels.json --system-config ${SYSTEM_CONFIG}
          --output ${GENERATED_DIR}/weights
  DEPENDS ${VISION_DIR}/kernels.json ${MODEL_SOURCE}/vision.py ${MODEL_SOURCE}/weights.py
          ${CMAKE_CURRENT_SOURCE_DIR}/quant/vision.py ${CMAKE_CURRENT_SOURCE_DIR}/trace/vision.py
          ${PACK_SOURCES} ${SYSTEM_CONFIG}
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
add_custom_target(omni-vision-weights DEPENDS ${GENERATED_DIR}/weights/vision-placement.json)
foreach(STAGE IN LISTS VISION_STAGES)
  set(TARGET ffn)
  if(STAGE MATCHES "^vision_(patch|attention)_")
    set(TARGET attention)
  endif()
  compile_graph(${STAGE} "${VISION_DIR}/${STAGE}.mlir" ${TARGET} kernel)
  compile_graph(${STAGE}-forward "${VISION_DIR}/${STAGE}-forward.mlir" ${TARGET} forward)
  list(APPEND OBJECTS ${STAGE}.o ${STAGE}-forward.o)
endforeach()
