set(TRACE_SUPPORTED TRUE PARENT_SCOPE)
set(MODEL_SOURCE "${REPO_ROOT}/stack/models/models/gemma4-e2b-it")
set(LLVM_OPT_LEVEL 2)
set(LLVM_PASSES "")
set(MATTR "+xbuckyball,+m,+D")
set(TRACE_ARGS)
set(TRACE_MODES)
set(TRACE_PASS)
if(CTRACE OR DTRACE)
  list(APPEND TRACE_ARGS --trace --trace-config ${CMAKE_CURRENT_SOURCE_DIR}/trace/trace.toml)
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
file(GLOB_RECURSE DESIGN_DEPENDS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/trace/*")
add_subdirectory(${MODEL_SOURCE} prepare)
set(KERNEL_PASSES
  -eliminate-empty-tensors -empty-tensor-to-alloc-tensor
  -convert-elementwise-to-linalg -convert-tensor-to-linalg
  -one-shot-bufferize=bufferize-function-boundaries
  -llvm-request-c-wrappers ${TRACE_PASS}
  -expand-strided-metadata -convert-linalg-to-loops
  -buffer-deallocation-simplification -bufferization-lower-deallocations
  -convert-vector-to-scf -lower-affine -convert-scf-to-cf -convert-cf-to-llvm
  -convert-vector-to-llvm -convert-arith-to-llvm -convert-math-to-llvm
  -convert-math-to-libm -convert-func-to-llvm -finalize-memref-to-llvm
  -reconcile-unrealized-casts)
set(OBJECTS)
foreach(PHASE prefill decode)
  compile_graph(forward_${PHASE} "${GENERATED_DIR}/forward_${PHASE}.mlir" attention kernel)
  compile_graph(subgraph_${PHASE} "${GENERATED_DIR}/subgraph0_${PHASE}.mlir" attention kernel)
  list(APPEND OBJECTS forward_${PHASE}.o subgraph_${PHASE}.o)
endforeach()
add_library(gemma4_kernels STATIC ${OBJECTS})
set_target_properties(gemma4_kernels PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(gemma4_kernels gemma4-model-build)
add_executable(gemma4-e2b-it-run ${MODEL_SOURCE}/main.cpp)
add_dependencies(gemma4-e2b-it-run model-loader)
target_include_directories(gemma4-e2b-it-run PRIVATE ${BUDDY_SOURCE}/frontend/Interfaces)
target_link_libraries(gemma4-e2b-it-run PRIVATE -static gemma4_kernels runtime)
add_custom_command(TARGET gemma4-e2b-it-run POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${ARTIFACT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:gemma4-e2b-it-run> ${ARTIFACT_DIR}/gemma4-e2b-it-run
  COMMAND ${PYTHON} -m stack.models.models.gemma4-e2b-it.pack
          --generated-dir ${GENERATED_DIR} --model-dir ${ARTIFACT_DIR}
          --chip ${CHIP} --rax-pack ${COMPILER_BUILD}/bin/rax-pack
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
set_property(TARGET gemma4-e2b-it-run APPEND PROPERTY LINK_DEPENDS
  ${MODEL_SOURCE}/pack.py ${REPO_ROOT}/stack/compiler/package.py)
