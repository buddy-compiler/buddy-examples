set(TRACE_SUPPORTED TRUE PARENT_SCOPE)
set(DESIGN_MODULE examples.models.poly.laya)
set(DESIGN_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(MODEL_SOURCE "${REPO_ROOT}/stack/models/models/laya")
set(LLVM_OPT_LEVEL 2 CACHE STRING "Backend optimization level")
set(LLVM_PASSES "function(sroa,early-cse,instcombine,simplifycfg,adce)")
set(MATTR "+xbuckyball,+m,+D")
option(CTRACE "Enable stage cycle trace" OFF)
option(DTRACE "Enable stage tensor trace" OFF)
file(GLOB_RECURSE DESIGN_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.py" "${CMAKE_CURRENT_SOURCE_DIR}/trace/*.json")
file(GLOB QUANT_FILES CONFIGURE_DEPENDS "${REPO_ROOT}/stack/compiler/quant/*.py")
list(APPEND DESIGN_FILES ${QUANT_FILES} ${REPO_ROOT}/stack/compiler/quant/mxfp8_embedding.py ${BUDDY_SOURCE}/frontend/Python/graph/transform/layout/convolution.py
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
  -convert-tensor-to-linalg -fuse-pointwise
  -outline-rvv-matmul -outline-rvv-kernels -fold-unit-transpose -canonicalize
  -canonicalize -cse
  -one-shot-bufferize=bufferize-function-boundaries -lower-rvv-kernels
  -lower-rvv-pointwise
  -lower-gather-rows -fuse-norm-quant-window
  -buffer-deallocation-pipeline=private-function-dynamic-ownership=false
  -convert-linalg-to-tile=quantized-only -convert-tile-to-buckyball
  -lower-buckyball-to-bank-ssa -assign-physical-banks
  -llvm-request-c-wrappers -lower-bank-ssa-to-intrinsics
  -expand-strided-metadata -convert-linalg-to-loops
  -lower-buckyball -convert-vector-to-scf=target-rank=0
  -lower-affine -canonicalize -cse
  -convert-scf-to-cf -convert-cf-to-llvm -convert-vector-to-llvm
  -buffer-deallocation-simplification
  -bufferization-lower-deallocations -convert-math-to-llvm
  -convert-math-to-libm -convert-arith-to-llvm -convert-func-to-llvm
  -finalize-memref-to-llvm -reconcile-unrealized-casts -canonicalize -cse
  -lower-tile-runtime -lower-ant-runtime -verify-npu-compute)
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
add_executable(laya-run ${MODEL_SOURCE}/main.cpp ${MODEL_SOURCE}/execution.cpp ${MODEL_SOURCE}/decision.cpp ${MODEL_SOURCE}/model.h)
add_dependencies(laya-run model-loader)
target_include_directories(laya-run PRIVATE ${ISA_DIR}/chip ${GENERATED_DIR} ${BUDDY_SOURCE}/frontend/Interfaces)
add_subdirectory("${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/rvv" rvv)
add_subdirectory("${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/mxfp8" mxfp8)
add_subdirectory("${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/mxquant" mxquant)
add_subdirectory("${REPO_ROOT}/examples/balls/mxmm/workloads/kernels" mxmm)
target_link_libraries(laya-run PRIVATE -static laya_kernels runtime_ant_tasks rvv_kernels_ant mxfp8_pack_ant mxquant_backend_ant mxmm_kernels_ant)
target_compile_options(laya-run PRIVATE -O${LLVM_OPT_LEVEL})
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
          --model-dir ${ARTIFACT_DIR} --run-config ${DESIGN_DIR}/configs/running-param.toml --rax-pack ${COMPILER_BUILD}/bin/rax-pack
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
set_property(TARGET laya-run APPEND PROPERTY LINK_DEPENDS ${MODEL_SOURCE}/pack.py ${MODEL_SOURCE}/inputs.py ${DESIGN_DIR}/configs/running-param.toml ${DESIGN_DIR}/trace/requests.json ${REPO_ROOT}/stack/compiler/package.py)
