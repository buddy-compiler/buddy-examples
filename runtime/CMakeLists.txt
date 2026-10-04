get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
add_library(runtime STATIC src/runtime.c src/workspace.c src/dma.c src/mlir.cpp)
target_compile_options(runtime PRIVATE -O2)
target_include_directories(runtime
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${RUNTIME_PARAMS}
          ${REPO_ROOT}/stack/compiler/thirdparty/buddy-mlir/llvm/mlir/include/mlir/ExecutionEngine)
target_include_directories(runtime PRIVATE ${REPO_ROOT}/bb-tests/workloads/lib)
target_compile_features(runtime PRIVATE c_std_11 cxx_std_17)

add_library(runtime_linux STATIC EXCLUDE_FROM_ALL src/linux_tasks.c src/workspace.c src/dma.c src/mlir.cpp)
target_compile_options(runtime_linux PRIVATE -O2)
get_target_property(RUNTIME_INCLUDES runtime INCLUDE_DIRECTORIES)
target_include_directories(runtime_linux PUBLIC ${RUNTIME_INCLUDES})
target_compile_features(runtime_linux PRIVATE c_std_11 cxx_std_17)
target_link_libraries(runtime_linux PUBLIC pthread)
