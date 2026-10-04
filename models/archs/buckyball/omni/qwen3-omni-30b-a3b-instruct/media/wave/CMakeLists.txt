set(WAVE_DIR "${GENERATED_DIR}/wave")
set(WAVE_STAGES)
foreach(COUNT 1 2 4 8 16 32 64 128)
  foreach(KIND attention dense norm up0 next0 up1 next1 input output)
    list(APPEND WAVE_STAGES wave_${KIND}_${COUNT})
  endforeach()
  foreach(INDEX 0 1 2 3)
    list(APPEND WAVE_STAGES wave_decoder${INDEX}_up_${COUNT})
    foreach(UNIT 0 1 2)
      list(APPEND WAVE_STAGES wave_decoder${INDEX}_res${UNIT}_${COUNT})
    endforeach()
  endforeach()
endforeach()
set(WAVE_FILES)
foreach(STAGE IN LISTS WAVE_STAGES)
  list(APPEND WAVE_FILES ${WAVE_DIR}/${STAGE}.mlir ${WAVE_DIR}/${STAGE}-forward.mlir)
endforeach()
add_custom_command(OUTPUT ${WAVE_FILES} ${WAVE_DIR}/kernels.json ${WAVE_DIR}/wave-parameters.h
  COMMAND ${PYTHON} -m ${DESIGN_MODULE}.trace.wave_kernels
          --checkpoint-config ${GENERATED_DIR}/checkpoint.json
          --compiler-build ${COMPILER_BUILD} --output ${WAVE_DIR}
  DEPENDS ${GENERATED_DIR}/checkpoint.json ${MODEL_SOURCE}/wave_conv.py ${MODEL_SOURCE}/wave_transformer.py
          ${MODEL_SOURCE}/projections.py ${MODEL_SOURCE}/weights.py ${GRAPH_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/trace/wave.py ${CMAKE_CURRENT_SOURCE_DIR}/trace/wave_kernels.py
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
add_custom_command(OUTPUT ${GENERATED_DIR}/weights/wave-placement.json
  COMMAND ${PYTHON} -m ${DESIGN_MODULE}.quant.wave
          --checkpoint-config ${GENERATED_DIR}/checkpoint.json --compiler-build ${COMPILER_BUILD}
          --kernels ${WAVE_DIR}/kernels.json --system-config ${SYSTEM_CONFIG}
          --output ${GENERATED_DIR}/weights
  DEPENDS ${WAVE_DIR}/kernels.json ${MODEL_SOURCE}/wave_conv.py ${MODEL_SOURCE}/wave_transformer.py
          ${MODEL_SOURCE}/weights.py ${PACK_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/quant/wave.py
          ${CMAKE_CURRENT_SOURCE_DIR}/trace/wave.py ${SYSTEM_CONFIG}
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
add_custom_target(omni-wave-weights DEPENDS ${GENERATED_DIR}/weights/wave-placement.json)
foreach(STAGE IN LISTS WAVE_STAGES)
  set(TARGET ffn)
  if(STAGE MATCHES "^wave_(attention|norm)_")
    set(TARGET attention)
  endif()
  compile_graph(${STAGE} "${WAVE_DIR}/${STAGE}.mlir" ${TARGET} kernel)
  compile_graph(${STAGE}-forward "${WAVE_DIR}/${STAGE}-forward.mlir" ${TARGET} forward)
  list(APPEND OBJECTS ${STAGE}.o ${STAGE}-forward.o)
endforeach()
