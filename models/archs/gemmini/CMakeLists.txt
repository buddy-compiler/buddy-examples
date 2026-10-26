set(GEMMINI_PLUGIN "${CMAKE_CURRENT_BINARY_DIR}/libgemmini-model-compiler.so")
add_custom_command(
  OUTPUT ${GEMMINI_PLUGIN}
  COMMAND "${TOOLCHAIN}/bin/g++" -std=c++17 -shared -fPIC -fno-rtti -I${BUDDY_SOURCE}/llvm/mlir/include -I${BUDDY_SOURCE}/llvm/llvm/include -I${BUDDY_SOURCE}/llvm/build/include -I${BUDDY_SOURCE}/llvm/build/tools/mlir/include ${CMAKE_CURRENT_SOURCE_DIR}/Int8Contractions.cpp -o ${GEMMINI_PLUGIN}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Int8Contractions.cpp
  VERBATIM
)
add_custom_target(gemmini-model-compiler DEPENDS ${GEMMINI_PLUGIN})
set(GEMMINI_OPT ${COMPILER_BUILD}/bin/buddy-opt --load-pass-plugin=${GEMMINI_PLUGIN})
add_custom_target(
  check-gemmini-model-compiler
  COMMAND
    ${PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/int8_contractions.py ${COMPILER_BUILD}/bin/buddy-opt ${GEMMINI_PLUGIN}
    ${BUDDY_SOURCE}/llvm/build/lib/libmlir_c_runner_utils.so
  DEPENDS ${GEMMINI_PLUGIN}
  VERBATIM
)

add_library(gemmini-runtime OBJECT EXCLUDE_FROM_ALL "${LIB_DIR}/CRunnerUtils.cpp" "${LIB_DIR}/native_trace.cpp"
  "${RUNTIME_DIR}/trace/trace.cpp"
)
target_include_directories(gemmini-runtime
  PRIVATE "${RUNTIME_DIR}/trace"
          "${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine"
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
