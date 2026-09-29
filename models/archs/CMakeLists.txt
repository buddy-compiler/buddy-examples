if(NOT DEFINED ARCH)
  message(FATAL_ERROR "Set ARCH")
endif()
add_subdirectory("${ARCH}")
