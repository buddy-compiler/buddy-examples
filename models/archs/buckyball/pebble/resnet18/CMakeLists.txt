set(TRACE_SUPPORTED TRUE PARENT_SCOPE)
set(DESIGN_MODULE examples.models.pebble.resnet18)
set(LLVM_OPT_LEVEL 2 CACHE STRING "Backend optimization level")
set(LLVM_PASSES "")
set(MODEL_SOURCE "${REPO_ROOT}/stack/models/models/resnet18")
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
  ${REPO_ROOT}/stack/compiler/quant/importer.py
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/quantization/activation.py
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/quantization/symmetric.py
  ${REPO_ROOT}/stack/compiler/quant/rax.py
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/layout/convolution.py
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/layout/matrix.py)
add_subdirectory(${MODEL_SOURCE} prepare)
set(FORWARD_PASSES -buffer-deallocation-simplification -convert-linalg-to-loops -eliminate-empty-tensors -llvm-request-c-wrappers -convert-math-to-llvm -convert-math-to-libm -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -expand-strided-metadata -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts)
set(KERNEL_PASSES
  -eliminate-empty-tensors
  -convert-elementwise-to-linalg
  -convert-tensor-to-linalg
  -one-shot-bufferize=bufferize-function-boundaries
  -convert-linalg-to-tile
  -convert-tile-to-buckyball
  -lower-buckyball-to-bank-ssa
  -assign-physical-banks
  -llvm-request-c-wrappers
  -lower-bank-ssa-to-intrinsics
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
  -reconcile-unrealized-casts)
compile_graph(forward "${GENERATED_DIR}/forward.mlir" pebble forward)
compile_graph(subgraph0 "${GENERATED_DIR}/subgraph0.mlir" pebble kernel)
add_library(resnet18_kernels STATIC forward.o subgraph0.o)
set_target_properties(resnet18_kernels PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(resnet18_kernels resnet18-model-build)
add_subdirectory("${REPO_ROOT}/stack/models/lib" image-runtime)
add_executable(resnet18-run ${MODEL_SOURCE}/main.cpp)
add_dependencies(resnet18-run model-loader)
target_include_directories(resnet18-run PRIVATE ${GENERATED_DIR}
  ${BUDDY_SOURCE}/frontend/Interfaces ${REPO_ROOT}/stack/models/lib)
target_link_libraries(resnet18-run PRIVATE -static resnet18_kernels runtime model_image)
add_custom_command(TARGET resnet18-run POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${ARTIFACT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:resnet18-run> ${ARTIFACT_DIR}/resnet18-run
  COMMAND ${PYTHON} ${MODEL_SOURCE}/pack.py --generated-dir ${GENERATED_DIR}
          --model-dir ${ARTIFACT_DIR} --rax-pack ${COMPILER_BUILD}/bin/rax-pack
          --arguments --backend buckyball
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)

set_property(TARGET resnet18-run APPEND PROPERTY LINK_DEPENDS ${MODEL_SOURCE}/pack.py ${REPO_ROOT}/stack/compiler/package.py)
