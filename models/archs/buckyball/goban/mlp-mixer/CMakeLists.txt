set(TRACE_SUPPORTED FALSE PARENT_SCOPE)
set(DESIGN_MODULE examples.models.goban.mlp-mixer)
set(LLVM_OPT_LEVEL 2 CACHE STRING "Backend optimization level")
set(LLVM_PASSES "")
set(MODEL_SOURCE "${REPO_ROOT}/stack/models/models/mlp-mixer")
set(KERNEL_GRAPHS)
foreach(INDEX RANGE 0 49)
  list(APPEND KERNEL_GRAPHS subgraph${INDEX} linear${INDEX})
endforeach()
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
set(BANK_SSA_PASS -lower-buckyball-to-bank-ssa)
if(DTRACE)
  set(BANK_SSA_PASS "--lower-buckyball-to-bank-ssa=trace-mega-stages=true")
endif()
file(GLOB_RECURSE DESIGN_FILES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/quant/*" "${CMAKE_CURRENT_SOURCE_DIR}/trace/*"
  "${CMAKE_CURRENT_SOURCE_DIR}/permute/*")
set(DESIGN_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py ${DESIGN_FILES}
  ${CMAKE_CURRENT_SOURCE_DIR}/lowering.py
  ${REPO_ROOT}/stack/compiler/quant/importer.py
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/quantization/activation.py
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/quantization/symmetric.py
  ${REPO_ROOT}/stack/compiler/quant/rax.py
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/layout/convolution.py
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/layout/matrix.py)
add_subdirectory(${MODEL_SOURCE} prepare)
set(FORWARD_PASSES -buffer-deallocation-pipeline -buffer-deallocation-simplification -convert-linalg-to-loops -eliminate-empty-tensors -llvm-request-c-wrappers -convert-math-to-llvm -convert-math-to-libm -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -expand-strided-metadata -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts)
set(KERNEL_PASSES
  -eliminate-empty-tensors
  -convert-elementwise-to-linalg
  -convert-tensor-to-linalg
  -one-shot-bufferize=bufferize-function-boundaries
  -buffer-deallocation-pipeline
  -convert-linalg-to-tile=quantized-only
  -convert-tile-to-buckyball
  ${BANK_SSA_PASS}
  -assign-physical-banks
  -llvm-request-c-wrappers
  -lower-bank-ssa-to-intrinsics
  ${TRACE_PASS}
  -expand-strided-metadata
  -convert-linalg-to-loops
  -lower-buckyball
  -canonicalize
  -cse
  -lower-affine
  -convert-scf-to-cf
  -convert-cf-to-llvm
  -buffer-deallocation-simplification
  -bufferization-lower-deallocations
  -convert-math-to-llvm
  -convert-math-to-libm
  -convert-arith-to-llvm
  -convert-func-to-llvm
  -finalize-memref-to-llvm
  -reconcile-unrealized-casts -lower-tile-runtime)
set(KERNEL_OBJECTS)
foreach(INDEX RANGE 0 49)
  compile_graph(linear${INDEX} "${GENERATED_DIR}/linear${INDEX}.mlir" goban forward)
  compile_graph(subgraph${INDEX} "${GENERATED_DIR}/subgraph${INDEX}.mlir" goban kernel)
  list(APPEND KERNEL_OBJECTS linear${INDEX}.o subgraph${INDEX}.o)
endforeach()
add_library(mixer_kernels STATIC ${KERNEL_OBJECTS})
set_target_properties(mixer_kernels PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(mixer_kernels mixer-model-build)
add_executable(mixer-run ${MODEL_SOURCE}/main.cpp)
add_dependencies(mixer-run model-loader)
target_include_directories(mixer-run PRIVATE ${GENERATED_DIR}
  ${BUDDY_SOURCE}/frontend/Interfaces ${BUDDY_SOURCE}/llvm/mlir/include ${REPO_ROOT}/stack/models/lib)
target_include_directories(mixer-run PRIVATE ${RUNTIME_PARAMS})
add_subdirectory("${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels" software)
target_link_libraries(mixer-run PRIVATE -static mixer_kernels runtime scalar_kernels)
add_custom_command(TARGET mixer-run POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${ARTIFACT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:mixer-run> ${ARTIFACT_DIR}/mixer-run
  COMMAND ${PYTHON} ${MODEL_SOURCE}/pack.py --generated-dir ${GENERATED_DIR}
          --model-dir ${ARTIFACT_DIR} --rax-pack ${COMPILER_BUILD}/bin/rax-pack
          --backend buckyball
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)

set_property(TARGET mixer-run APPEND PROPERTY LINK_DEPENDS ${MODEL_SOURCE}/pack.py ${REPO_ROOT}/stack/compiler/package.py ${GENERATED_DIR}/model.json)
