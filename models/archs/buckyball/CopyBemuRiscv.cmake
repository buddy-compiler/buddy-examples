if(NOT BEMU_RUNTIME_LIBRARY OR NOT OUTPUT_LIBRARY)
  message(FATAL_ERROR
    "BEMU_RUNTIME_LIBRARY and OUTPUT_LIBRARY are required")
endif()

get_filename_component(_release_dir "${BEMU_RUNTIME_LIBRARY}" DIRECTORY)
file(GLOB _riscv_libraries LIST_DIRECTORIES false
  "${_release_dir}/build/bemu-*/out/spike_install/lib/libriscv.so")
list(LENGTH _riscv_libraries _riscv_library_count)
if(_riscv_library_count EQUAL 0)
  message(FATAL_ERROR
    "Cannot find the Spike runtime library below ${_release_dir}/build")
endif()

list(GET _riscv_libraries 0 _riscv_library)
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "${_riscv_library}" "${OUTPUT_LIBRARY}"
  COMMAND_ERROR_IS_FATAL ANY)
