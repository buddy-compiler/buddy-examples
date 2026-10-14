include(ExternalProject)
ExternalProject_Add(
  model-loader
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/loader"
  BINARY_DIR "${CMAKE_BINARY_DIR}/loader"
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
  BUILD_ALWAYS TRUE
  INSTALL_COMMAND ""
)
