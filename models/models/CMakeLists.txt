if(NOT DEFINED MODEL)
  message(FATAL_ERROR "Set MODEL")
endif()
add_subdirectory("${MODEL}")
