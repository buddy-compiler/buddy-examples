find_package(PNG REQUIRED)
add_library(nv-runtime OBJECT EXCLUDE_FROM_ALL "${MODEL_LIB_DIR}/CRunnerUtils.cpp" "${MODEL_LIB_DIR}/native_trace.cpp"
  "${RUNTIME_DIR}/src/trace.cpp"
)
target_include_directories(nv-runtime PRIVATE "${RUNTIME_DIR}/include"
                                              "${RUNTIME_DIR}/src"
                                              "${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine"
)
set_target_properties(nv-runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(bert-base-uncased-emotion)
add_subdirectory(deepseek-r1-distill-qwen-1.5b)
add_subdirectory(gemma4-e2b-it)
add_subdirectory(lenet)
add_subdirectory(minimax-h3-fl2va)
add_subdirectory(minimax-h3-ref2va)
add_subdirectory(mobilenet-v3-small)
add_subdirectory(qwen3-8b)
add_subdirectory(resnet18)
add_subdirectory(stable-diffusion)
add_subdirectory(whisper)
add_subdirectory(yolo26n)
