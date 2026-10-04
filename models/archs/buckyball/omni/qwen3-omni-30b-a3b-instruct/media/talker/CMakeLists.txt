set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/configs/compiler-param.toml")
set(TALKER_DIR "${GENERATED_DIR}/talker")
set(TALKER_STAGES talker_output predictor_output)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/configs/compiler-param.toml" CACHE_LENGTH REGEX "^cache_length = [0-9]+$")
string(REPLACE "cache_length = " "" TALKER_CAPACITY "${CACHE_LENGTH}")
set(TALKER_CACHE_BUCKETS)
set(CAPACITY 16)
while(CAPACITY LESS TALKER_CAPACITY)
  list(APPEND TALKER_CACHE_BUCKETS ${CAPACITY})
  math(EXPR CAPACITY "${CAPACITY} * 2")
endwhile()
list(APPEND TALKER_CACHE_BUCKETS ${TALKER_CAPACITY})
foreach(PHASE prefill decode)
  foreach(CAPACITY IN LISTS TALKER_CACHE_BUCKETS)
    list(APPEND TALKER_STAGES talker_${PHASE}_attention_${CAPACITY})
  endforeach()
  foreach(KIND router shared norm text_projection hidden_projection)
    list(APPEND TALKER_STAGES talker_${PHASE}_${KIND})
  endforeach()
  foreach(KIND attention dense norm)
    list(APPEND TALKER_STAGES predictor_${PHASE}_${KIND})
  endforeach()
endforeach()
foreach(COUNT 1 2 4 8 16)
  list(APPEND TALKER_STAGES talker_expert_${COUNT})
endforeach()
set(TALKER_FILES)
foreach(STAGE IN LISTS TALKER_STAGES)
  list(APPEND TALKER_FILES ${TALKER_DIR}/${STAGE}.mlir ${TALKER_DIR}/${STAGE}-forward.mlir)
endforeach()
add_custom_command(OUTPUT ${TALKER_FILES} ${TALKER_DIR}/kernels.json ${TALKER_DIR}/talker-parameters.h
  COMMAND ${PYTHON} -m ${DESIGN_MODULE}.trace.talker_kernels
          --checkpoint-config ${GENERATED_DIR}/checkpoint.json
          --compiler-build ${COMPILER_BUILD} --output ${TALKER_DIR}
  DEPENDS ${GENERATED_DIR}/checkpoint.json ${MODEL_SOURCES} ${MODEL_SOURCE}/talker.py ${GRAPH_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/trace/talker.py ${CMAKE_CURRENT_SOURCE_DIR}/trace/talker_kernels.py
          ${CMAKE_CURRENT_SOURCE_DIR}/configs/compiler-param.toml
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
add_custom_command(OUTPUT ${GENERATED_DIR}/weights/talker-placement.json
  COMMAND ${PYTHON} -m ${DESIGN_MODULE}.quant.talker
          --checkpoint-config ${GENERATED_DIR}/checkpoint.json --compiler-build ${COMPILER_BUILD}
          --kernels ${TALKER_DIR}/kernels.json --system-config ${SYSTEM_CONFIG}
          --output ${GENERATED_DIR}/weights
  DEPENDS ${TALKER_DIR}/kernels.json ${MODEL_SOURCES} ${MODEL_SOURCE}/talker.py ${PACK_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/quant/talker.py ${CMAKE_CURRENT_SOURCE_DIR}/trace/talker.py ${SYSTEM_CONFIG}
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
add_custom_target(omni-talker-weights DEPENDS ${GENERATED_DIR}/weights/talker-placement.json)
foreach(STAGE IN LISTS TALKER_STAGES)
  set(TARGET ffn)
  if(STAGE MATCHES "_(attention(_[0-9]+)?|router|norm)$")
    set(TARGET attention)
  endif()
  compile_graph(${STAGE} "${TALKER_DIR}/${STAGE}.mlir" ${TARGET} kernel)
  compile_graph(${STAGE}-forward "${TALKER_DIR}/${STAGE}-forward.mlir" ${TARGET} forward)
  list(APPEND OBJECTS ${STAGE}.o ${STAGE}-forward.o)
endforeach()
