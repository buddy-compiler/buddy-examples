add_library(gemmini-runtime OBJECT EXCLUDE_FROM_ALL "${MODEL_LIB_DIR}/CRunnerUtils.cpp"
  "${MODEL_LIB_DIR}/native_trace.cpp" "${RUNTIME_DIR}/src/trace.cpp"
)
target_include_directories(gemmini-runtime PRIVATE "${RUNTIME_DIR}/include"
                                                   "${RUNTIME_DIR}/src"
                                                   "${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine"
)

add_subdirectory("${MODEL_LIB_DIR}" image-runtime EXCLUDE_FROM_ALL)

add_subdirectory(bert-base-uncased-emotion)
add_subdirectory(deepseek-r1-distill-qwen-1.5b)
add_subdirectory(gemma4-e2b-it)
add_subdirectory(lenet)
add_subdirectory(mobilenet-v3-small)
add_subdirectory(qwen3-8b)
add_subdirectory(resnet18)
add_subdirectory(stable-diffusion)
add_subdirectory(yolo26n)
