set(AUDIO_DIR "${GENERATED_DIR}/audio")
set(AUDIO_STAGES audio_conv1 audio_conv2 audio_conv3 audio_projection)
foreach(COUNT 1 4 16 64 128)
  foreach(KIND attention mlp output)
    list(APPEND AUDIO_STAGES audio_${KIND}_${COUNT})
  endforeach()
endforeach()
set(AUDIO_FILES)
foreach(STAGE IN LISTS AUDIO_STAGES)
  list(APPEND AUDIO_FILES ${AUDIO_DIR}/${STAGE}.mlir ${AUDIO_DIR}/${STAGE}-forward.mlir)
endforeach()
add_custom_command(OUTPUT ${AUDIO_FILES} ${AUDIO_DIR}/kernels.json ${AUDIO_DIR}/audio-parameters.h
  COMMAND ${PYTHON} -m ${DESIGN_MODULE}.trace.audio_kernels
          --checkpoint-config ${GENERATED_DIR}/checkpoint.json
          --compiler-build ${COMPILER_BUILD} --output ${AUDIO_DIR}
  DEPENDS ${GENERATED_DIR}/checkpoint.json ${MODEL_SOURCE}/audio.py ${MODEL_SOURCE}/weights.py
          ${CMAKE_CURRENT_SOURCE_DIR}/trace/audio.py ${CMAKE_CURRENT_SOURCE_DIR}/trace/audio_kernels.py
          ${GRAPH_SOURCES}
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
add_custom_command(OUTPUT ${GENERATED_DIR}/weights/audio-placement.json
  COMMAND ${PYTHON} -m ${DESIGN_MODULE}.quant.audio
          --checkpoint-config ${GENERATED_DIR}/checkpoint.json --compiler-build ${COMPILER_BUILD}
          --kernels ${AUDIO_DIR}/kernels.json --system-config ${SYSTEM_CONFIG}
          --output ${GENERATED_DIR}/weights
  DEPENDS ${AUDIO_DIR}/kernels.json ${MODEL_SOURCE}/audio.py ${MODEL_SOURCE}/weights.py
          ${CMAKE_CURRENT_SOURCE_DIR}/quant/audio.py ${CMAKE_CURRENT_SOURCE_DIR}/trace/audio.py
          ${PACK_SOURCES} ${SYSTEM_CONFIG}
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
add_custom_target(omni-audio-weights DEPENDS ${GENERATED_DIR}/weights/audio-placement.json)
foreach(STAGE IN LISTS AUDIO_STAGES)
  set(TARGET attention)
  if(STAGE MATCHES "^audio_(mlp|output)_")
    set(TARGET ffn)
  endif()
  compile_graph(${STAGE} "${AUDIO_DIR}/${STAGE}.mlir" ${TARGET} kernel)
  compile_graph(${STAGE}-forward "${AUDIO_DIR}/${STAGE}-forward.mlir" ${TARGET} forward)
  list(APPEND OBJECTS ${STAGE}.o ${STAGE}-forward.o)
endforeach()
