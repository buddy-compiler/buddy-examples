add_library(gemmini-runtime OBJECT EXCLUDE_FROM_ALL "${LIB_DIR}/CRunnerUtils.cpp" "${LIB_DIR}/native_trace.cpp"
  "${RUNTIME_DIR}/trace/trace.cpp"
)
target_include_directories(gemmini-runtime
  PRIVATE "${RUNTIME_DIR}/trace"
          "${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine"
)
# The runners are shared libraries.
set_target_properties(gemmini-runtime
  PROPERTIES POSITION_INDEPENDENT_CODE ON
)

add_subdirectory("${LIB_DIR}" image-runtime EXCLUDE_FROM_ALL)

if(MODEL)
  add_subdirectory("${MODEL}")
else()
  add_subdirectory(alexnet)
  add_subdirectory(bert-base-uncased-emotion)
  add_subdirectory(lenet)
  add_subdirectory(mobilenet-v3-small)
  add_subdirectory(resnet18)
  add_subdirectory(yolo26n)
endif()
