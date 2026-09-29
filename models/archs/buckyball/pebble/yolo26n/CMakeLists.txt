set(TRACE_SUPPORTED TRUE PARENT_SCOPE)
set(DESIGN_MODULE examples.models.pebble.yolo26n)
set(LLVM_OPT_LEVEL 2 CACHE STRING "Backend optimization level")
set(LLVM_PASSES "")
set(MODEL_SOURCE "${REPO_ROOT}/stack/models/models/yolo26n")
set(CALIBRATION_IMAGES
  ${REPO_ROOT}/stack/models/models/resnet18/images/dog.png
  ${REPO_ROOT}/stack/models/models/resnet18/images/ice-cream.png
  ${REPO_ROOT}/stack/models/models/resnet18/images/traffic-light.png
  ${REPO_ROOT}/stack/models/models/resnet18/images/kite.png
  ${REPO_ROOT}/stack/models/models/resnet18/images/curtain.png)
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
set(FORWARD_PASSES -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -convert-elementwise-to-linalg -one-shot-bufferize=bufferize-function-boundaries -expand-strided-metadata -convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -llvm-request-c-wrappers -buffer-deallocation-simplification -bufferization-lower-deallocations -convert-math-to-llvm -convert-math-to-libm -convert-arith-to-llvm -convert-func-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts)
set(KERNEL_PASSES
  -eliminate-empty-tensors
  -convert-elementwise-to-linalg
  -convert-tensor-to-linalg
  -one-shot-bufferize=bufferize-function-boundaries
  -convert-linalg-to-tile
  -convert-tile-to-buckyball
  ${BANK_SSA_PASS}
  -assign-physical-banks
  -lower-bank-ssa-to-intrinsics
  ${TRACE_PASS}
  -lower-buckyball
  -canonicalize
  -cse
  --mlir-disable-threading
  -llvm-request-c-wrappers
  -expand-strided-metadata
  -convert-linalg-to-loops
  -lower-affine
  -convert-scf-to-cf=allow-pattern-rollback=false
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
add_library(yolo26_kernels STATIC forward.o subgraph0.o)
set_target_properties(yolo26_kernels PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(yolo26_kernels yolo26-model-build)
add_subdirectory("${REPO_ROOT}/stack/models/lib" image-runtime)
add_executable(yolo26-run ${MODEL_SOURCE}/main.cpp)
add_dependencies(yolo26-run model-loader)
target_include_directories(yolo26-run PRIVATE ${GENERATED_DIR}
  ${BUDDY_SOURCE}/frontend/Interfaces ${REPO_ROOT}/stack/models/lib)
target_link_libraries(yolo26-run PRIVATE -static yolo26_kernels runtime model_image)
add_custom_command(TARGET yolo26-run POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${ARTIFACT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:yolo26-run> ${ARTIFACT_DIR}/yolo26-run
  COMMAND ${PYTHON} ${MODEL_SOURCE}/pack.py --generated-dir ${GENERATED_DIR}
          --model-dir ${ARTIFACT_DIR} --rax-pack ${COMPILER_BUILD}/bin/rax-pack
          --arguments --backend buckyball
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)

set_property(TARGET yolo26-run APPEND PROPERTY LINK_DEPENDS ${MODEL_SOURCE}/pack.py ${REPO_ROOT}/stack/compiler/package.py)
