set(TRACE_SUPPORTED TRUE PARENT_SCOPE)
set(DESIGN_MODULE examples.models.toy.lenet)
set(LLVM_OPT_LEVEL 2 CACHE STRING "Backend optimization level")
set(LLVM_PASSES "")
set(MODEL_SOURCE "${REPO_ROOT}/stack/models/models/lenet")
set(MATTR "+xbuckyball,+m,+D")
option(CTRACE "Enable cycle trace" OFF)
option(DTRACE "Enable tensor data trace" OFF)
set(TRACE_ARGS)
set(TRACE_MODES)
set(TRACE_PASS)
if(CTRACE OR DTRACE)
  list(APPEND TRACE_ARGS --trace)
endif()
if(CTRACE)
  list(APPEND TRACE_MODES cycle-trace)
endif()
if(DTRACE)
  list(APPEND TRACE_MODES tensor-trace)
endif()
if(TRACE_MODES)
  list(JOIN TRACE_MODES "," TRACE_MODES)
  set(TRACE_PASS "-convert-trace-to-llvm=${TRACE_MODES}")
endif()
file(GLOB_RECURSE DESIGN_FILES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/quant/*" "${CMAKE_CURRENT_SOURCE_DIR}/trace/*"
  "${CMAKE_CURRENT_SOURCE_DIR}/permute/*")
set(DESIGN_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py ${DESIGN_FILES}
  ${REPO_ROOT}/examples/balls/gemmini/compiler/python/quant.py
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/quantization/activation.py
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/quantization/symmetric.py
  ${REPO_ROOT}/stack/compiler/quant/rax.py
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/layout/convolution.py
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/layout/matrix.py)
add_subdirectory(${MODEL_SOURCE} prepare)
set(FORWARD_PASSES -buffer-deallocation-simplification -convert-linalg-to-loops -eliminate-empty-tensors -llvm-request-c-wrappers -convert-math-to-llvm -convert-math-to-libm -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -expand-strided-metadata -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts)
set(KERNEL_PASSES
  -eliminate-empty-tensors -one-shot-bufferize=bufferize-function-boundaries
  -lower-buckyball-to-bank-ssa -llvm-request-c-wrappers
  ${TRACE_PASS}
  -lower-bank-ssa-to-intrinsics -lower-buckyball -convert-linalg-to-loops
  -expand-strided-metadata -lower-affine -convert-scf-to-cf
  -convert-cf-to-llvm -buffer-deallocation-simplification
  -bufferization-lower-deallocations -convert-math-to-llvm -convert-math-to-libm
  -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm
  -reconcile-unrealized-casts)
compile_graph(forward "${GENERATED_DIR}/forward.mlir" toy forward)
compile_graph(subgraph0 "${GENERATED_DIR}/subgraph0.mlir" toy kernel)
add_library(lenet_kernels STATIC forward.o subgraph0.o)
set_target_properties(lenet_kernels PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(lenet_kernels lenet-model-build)
add_subdirectory("${REPO_ROOT}/stack/models/lib" image-runtime)
add_library(gemmini_runtime STATIC ${REPO_ROOT}/examples/balls/gemmini/compiler/runtime/conv.cpp)
target_include_directories(gemmini_runtime PRIVATE ${RUNTIME_PARAMS}
  ${REPO_ROOT}/stack/runtime/include ${REPO_ROOT}/bb-tests/workloads/lib
  ${REPO_ROOT}/examples/balls/gemmini/workloads
  ${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine)
add_executable(lenet-run ${MODEL_SOURCE}/main.cpp)
add_dependencies(lenet-run model-loader)
target_include_directories(lenet-run PRIVATE ${GENERATED_DIR}
  ${BUDDY_SOURCE}/frontend/Interfaces ${REPO_ROOT}/stack/models/lib)
target_link_libraries(lenet-run PRIVATE -static lenet_kernels gemmini_runtime runtime model_image)
add_custom_command(TARGET lenet-run POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${ARTIFACT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:lenet-run> ${ARTIFACT_DIR}/lenet-run
  COMMAND ${PYTHON} ${MODEL_SOURCE}/pack.py --generated-dir ${GENERATED_DIR}
          --model-dir ${ARTIFACT_DIR} --rax-pack ${COMPILER_BUILD}/bin/rax-pack
          --arguments=--image --backend buckyball
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)

set_property(TARGET lenet-run APPEND PROPERTY LINK_DEPENDS ${MODEL_SOURCE}/pack.py ${REPO_ROOT}/stack/compiler/package.py)
