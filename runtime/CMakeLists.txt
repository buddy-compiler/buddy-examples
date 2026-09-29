get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
add_library(runtime STATIC src/runtime.c src/dma.c src/mlir.cpp src/mxfp8.cpp)
target_compile_options(runtime PRIVATE -O2)
target_include_directories(runtime
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${RUNTIME_PARAMS}
          ${REPO_ROOT}/stack/compiler/thirdparty/buddy-mlir/llvm/mlir/include/mlir/ExecutionEngine)
target_compile_features(runtime PRIVATE c_std_11 cxx_std_17)
