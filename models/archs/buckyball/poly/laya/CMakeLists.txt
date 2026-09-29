set(TRACE_SUPPORTED TRUE PARENT_SCOPE)
set(DESIGN_MODULE examples.models.poly.laya)
set(MODEL_SOURCE "${REPO_ROOT}/stack/models/models/laya")
set(LLVM_OPT_LEVEL 2 CACHE STRING "Backend optimization level")
set(LLVM_PASSES "function(sroa,early-cse,instcombine,simplifycfg,adce)")
set(MATTR "+xbuckyball,+m,+D")
option(CTRACE "Enable stage cycle trace" OFF)
option(DTRACE "Enable stage tensor trace" OFF)
file(GLOB_RECURSE DESIGN_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.py" "${CMAKE_CURRENT_SOURCE_DIR}/trace/*.json")
file(GLOB QUANT_FILES CONFIGURE_DEPENDS "${REPO_ROOT}/stack/compiler/quant/*.py")
list(APPEND DESIGN_FILES ${QUANT_FILES} ${BUDDY_SOURCE}/frontend/Python/graph/transform/layout/convolution.py
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/layout/matrix.py)
add_subdirectory(${MODEL_SOURCE} prepare)
set(FORWARD_PASSES
  -buffer-deallocation-simplification -convert-linalg-to-loops
  -eliminate-empty-tensors -llvm-request-c-wrappers
  -convert-math-to-llvm -convert-math-to-libm -lower-affine
  -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
  -expand-strided-metadata -finalize-memref-to-llvm
  -convert-func-to-llvm -reconcile-unrealized-casts -canonicalize -cse -lower-tile-runtime)
set(KERNEL_PASSES
  -eliminate-empty-tensors -convert-elementwise-to-linalg
  -convert-tensor-to-linalg -one-shot-bufferize=bufferize-function-boundaries
  -convert-linalg-to-tile=quantized-only
  -convert-tile-to-buckyball
  -lower-buckyball-to-bank-ssa -assign-physical-banks
  -llvm-request-c-wrappers -lower-bank-ssa-to-intrinsics
  -expand-strided-metadata -convert-linalg-to-loops
  -lower-buckyball -lower-affine -canonicalize -cse -convert-scf-to-cf
  -convert-cf-to-llvm -buffer-deallocation-simplification
  -bufferization-lower-deallocations -convert-math-to-llvm
  -convert-math-to-libm -convert-arith-to-llvm -convert-func-to-llvm
  -finalize-memref-to-llvm -reconcile-unrealized-casts -canonicalize -cse -lower-tile-runtime)
set(OBJECTS)
foreach(STAGE IN LISTS STAGES)
  if(STAGE MATCHES "ffn" OR STAGE STREQUAL "scorer" OR STAGE STREQUAL "action")
    set(TARGET ffn)
  else()
    set(TARGET attention)
  endif()
  compile_graph(${STAGE} "${GENERATED_DIR}/${STAGE}.mlir" ${TARGET} kernel)
  compile_graph(${STAGE}-forward "${GENERATED_DIR}/${STAGE}-forward.mlir" ${TARGET} forward)
  list(APPEND OBJECTS ${STAGE}.o ${STAGE}-forward.o)
endforeach()
add_library(laya_kernels STATIC ${OBJECTS})
set_target_properties(laya_kernels PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(laya_kernels laya-model-build)
add_executable(laya-run ${MODEL_SOURCE}/main.cpp)
add_dependencies(laya-run model-loader)
target_include_directories(laya-run PRIVATE ${GENERATED_DIR} ${BUDDY_SOURCE}/frontend/Interfaces)
target_link_libraries(laya-run PRIVATE -static laya_kernels runtime)
target_compile_options(laya-run PRIVATE -O${LLVM_OPT_LEVEL})
target_compile_options(runtime PRIVATE -O${LLVM_OPT_LEVEL})
if(CTRACE)
  target_compile_definitions(laya-run PRIVATE CTRACE)
endif()
if(DTRACE)
  target_compile_definitions(laya-run PRIVATE DTRACE)
endif()
add_custom_command(TARGET laya-run POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${ARTIFACT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:laya-run> ${ARTIFACT_DIR}/laya-run
  COMMAND ${PYTHON} -m stack.models.models.laya.pack --generated-dir ${GENERATED_DIR}
          --model-dir ${ARTIFACT_DIR} --rax-pack ${COMPILER_BUILD}/bin/rax-pack
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
set_property(TARGET laya-run APPEND PROPERTY LINK_DEPENDS ${MODEL_SOURCE}/pack.py ${REPO_ROOT}/stack/compiler/package.py)
