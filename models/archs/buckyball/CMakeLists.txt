if(NOT DEFINED CHIP OR NOT DEFINED MODEL)
  message(FATAL_ERROR "Set CHIP and MODEL")
endif()
add_subdirectory("${CHIP}/${MODEL}")
