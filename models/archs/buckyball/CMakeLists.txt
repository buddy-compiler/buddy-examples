get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
set(DESIGNS_DIR "${REPO_ROOT}/examples/designs")

set(DESIGN_PB "${DESIGNS_DIR}/${DESIGN}/output/generated/design.pb")
# Everything this build needs from design.pb: ISA headers and the DESIGN_* lists.
execute_process(
  COMMAND
    ${PYTHON} "${REPO_ROOT}/stack/scripts/design.py" --design-pb "${DESIGN_PB}" --target "${BUCKYBALL_SIM_TARGET}" --out
    "${CMAKE_BINARY_DIR}/generated"
  COMMAND_ERROR_IS_FATAL ANY
)
include("${CMAKE_BINARY_DIR}/generated/design.cmake")
set(ISA_DIR "${CMAKE_BINARY_DIR}/generated/isa")
# DMA helpers are sized by the target with the fewest bank lines.
list(GET DESIGN_RUNTIME_TARGETS 0 PARAMS_TARGET)
set(RUNTIME_PARAMS "${ISA_DIR}/${PARAMS_TARGET}")
set(COMPILER_BUILD "${DESIGN_COMPILER}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${DESIGN_PB}" "${REPO_ROOT}/stack/scripts/design.py")
include_directories("${RUNTIME_DIR}/scheduler" "${RUNTIME_DIR}/services" "${ISA_DIR}/design")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# runtime, runtime-linux and engine: built by the device stack. Kernels and
# their kernels.rename: built by the thread stacks, in DESIGN_THREAD_BUILD_<thread>.
include("${DESIGN_DEVICE_BUILD}/device.cmake")

add_subdirectory("${LIB_DIR}" image-runtime EXCLUDE_FROM_ALL)

if(MODEL)
  add_subdirectory("${DESIGN}/${MODEL}")
else()
  add_subdirectory("${DESIGN}")
endif()
