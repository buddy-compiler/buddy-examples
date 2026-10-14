if(CHIP STREQUAL "pebble")
  set(CHIP_CONFIG_DIR "${CHIPS_DIR}/pebble/configs")
elseif(CHIP STREQUAL "poly")
  set(CHIP_CONFIG_DIR "${CHIPS_DIR}/poly/configs")
elseif(CHIP STREQUAL "goban")
  set(CHIP_CONFIG_DIR "${CHIPS_DIR}/goban/configs")
elseif(CHIP STREQUAL "omni")
  set(CHIP_CONFIG_DIR "${CHIPS_DIR}/omni/configs")
elseif(CHIP STREQUAL "toy")
  set(CHIP_CONFIG_DIR "${CHIPS_DIR}/toy/configs")
else()
  message(FATAL_ERROR "Unknown chip: ${CHIP}")
endif()
set(CHIP_PB "${CHIP_CONFIG_DIR}/generated/chip.pb")
set(ISA_DIR "${CMAKE_BINARY_DIR}/isa")
execute_process(
  COMMAND
    "${PYTHON}" "${CHIP_TARGETS}" --repo "${REPO_ROOT}" --chip-pb "${CHIP_PB}" --isa-dir "${ISA_DIR}" --print-targets
  OUTPUT_VARIABLE TARGETS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
string(REPLACE "\n" ";" TARGETS "${TARGETS}")
execute_process(
  COMMAND "${PYTHON}" "${CHIP_TARGETS}" --repo "${REPO_ROOT}" --chip-pb "${CHIP_PB}" --print-compiler-dirs
  OUTPUT_VARIABLE COMPILER_ROWS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
string(REPLACE "\n" ";" COMPILER_ROWS "${COMPILER_ROWS}")
foreach(ROW IN LISTS COMPILER_ROWS)
  string(REPLACE "|" ";" FIELDS "${ROW}")
  list(GET FIELDS 0 PROFILE)
  list(GET FIELDS 1 COMPILER_BUILD_${PROFILE})
endforeach()
execute_process(
  COMMAND "${PYTHON}" "${CHIP_TARGETS}" --repo "${REPO_ROOT}" --chip-pb "${CHIP_PB}" --print-runtime-targets
  OUTPUT_VARIABLE RUNTIME_TARGETS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
string(REPLACE "\n" ";" RUNTIME_TARGETS "${RUNTIME_TARGETS}")
list(GET RUNTIME_TARGETS 0 PARAMS_TARGET)
file(STRINGS "${ISA_DIR}/${PARAMS_TARGET}/params.h" BANK_LINES_DECL REGEX "^#define BANK_LINES ")
string(REGEX REPLACE "^#define BANK_LINES ([0-9]+)$" "\\1" RUNTIME_BANK_LINES "${BANK_LINES_DECL}")
file(STRINGS "${ISA_DIR}/${PARAMS_TARGET}/params.h" MEMORY_PARAMS REGEX "^#define (BANK_WIDTH|VIRTUAL_BANK_NUM) ")
foreach(PROFILE IN LISTS RUNTIME_TARGETS)
  file(STRINGS "${ISA_DIR}/${PROFILE}/params.h" PARAMS REGEX "^#define (BANK_WIDTH|VIRTUAL_BANK_NUM) ")
  if(NOT PARAMS STREQUAL MEMORY_PARAMS)
    message(FATAL_ERROR "DMA runtime requires matching bank width and virtual bank count")
  endif()
  file(STRINGS "${ISA_DIR}/${PROFILE}/params.h" BANK_LINES_DECL REGEX "^#define BANK_LINES ")
  string(REGEX REPLACE "^#define BANK_LINES ([0-9]+)$" "\\1" PROFILE_BANK_LINES "${BANK_LINES_DECL}")
  if(PROFILE_BANK_LINES LESS RUNTIME_BANK_LINES)
    set(PARAMS_TARGET "${PROFILE}")
    set(RUNTIME_BANK_LINES "${PROFILE_BANK_LINES}")
  endif()
endforeach()
set(RUNTIME_PARAMS "${ISA_DIR}/${PARAMS_TARGET}")
set(COMPILER_BUILD "${COMPILER_BUILD_${PARAMS_TARGET}}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CHIP_PB}")
include_directories("${RUNTIME_DIR}/include")
add_subdirectory("${RUNTIME_DIR}" "${CMAKE_BINARY_DIR}/runtime")
add_subdirectory("${RUNTIME_DIR}/threads" "${CMAKE_BINARY_DIR}/thread-runtime")

add_subdirectory("${MODEL_LIB_DIR}" image-runtime EXCLUDE_FROM_ALL)
add_subdirectory("${SOFTWARE_LIB_DIR}" software EXCLUDE_FROM_ALL)
set(RVV_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/rvv")
add_subdirectory("${SOFTWARE_LIB_DIR}/rvv" "${RVV_BUILD_DIR}" EXCLUDE_FROM_ALL)
add_subdirectory("${SOFTWARE_LIB_DIR}/mxfp8" mxfp8 EXCLUDE_FROM_ALL)

if(CHIP STREQUAL "pebble")
  add_subdirectory(pebble)
elseif(CHIP STREQUAL "poly")
  add_subdirectory(poly)
elseif(CHIP STREQUAL "goban")
  add_subdirectory(goban)
elseif(CHIP STREQUAL "omni")
  add_subdirectory(omni)
elseif(CHIP STREQUAL "toy")
  add_subdirectory(toy)
else()
  message(FATAL_ERROR "Unknown chip: ${CHIP}")
endif()
