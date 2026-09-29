set(DESIGN_MODULE examples.models.omni.qwen3-omni-30b-a3b-instruct)
set(MODEL_SOURCE "${REPO_ROOT}/stack/models/models/qwen3-omni-30b-a3b-instruct")
set(SYSTEM_CONFIG "${REPO_ROOT}/examples/chips/omni/configs/system.toml")
set(KERNEL_DIR "${GENERATED_DIR}/thinker")
set(LLVM_OPT_LEVEL 2)
set(LLVM_PASSES "default<O2>")
set(MATTR "+xbuckyball,+m,+D")
set(STAGES prefill_attention decode_attention prefill_router decode_router
  prefill_embedding decode_embedding prefill_norm decode_norm expert output)
set(KERNEL_FILES)
foreach(STAGE IN LISTS STAGES)
  list(APPEND KERNEL_FILES ${KERNEL_DIR}/${STAGE}.mlir ${KERNEL_DIR}/${STAGE}-forward.mlir)
endforeach()
file(GLOB_RECURSE DESIGN_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/trace/*.py" "${CMAKE_CURRENT_SOURCE_DIR}/permute/*.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/quant/*.py")
list(APPEND DESIGN_SOURCES ${REPO_ROOT}/examples/balls/smatmul/compiler/python/fp32.py)
set(MODEL_SOURCES ${MODEL_SOURCE}/attention.py ${MODEL_SOURCE}/moe.py ${MODEL_SOURCE}/projections.py)
add_custom_command(OUTPUT ${GENERATED_DIR}/checkpoint.json
  COMMAND ${PYTHON} -m stack.models.models.qwen3-omni-30b-a3b-instruct.prepare --output-dir ${GENERATED_DIR}
  DEPENDS ${MODEL_SOURCE}/prepare.py WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
add_custom_command(OUTPUT ${KERNEL_FILES} ${KERNEL_DIR}/kernels.json ${KERNEL_DIR}/model-parameters.h
  COMMAND ${PYTHON} -m ${DESIGN_MODULE}.trace.kernels
          --checkpoint-config ${GENERATED_DIR}/checkpoint.json
          --compiler-build ${COMPILER_BUILD} --output ${KERNEL_DIR} --system-config ${SYSTEM_CONFIG}
  DEPENDS ${GENERATED_DIR}/checkpoint.json ${MODEL_SOURCES} ${DESIGN_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/configs/compiler-param.toml ${SYSTEM_CONFIG}
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
add_custom_command(OUTPUT ${GENERATED_DIR}/weights/placements.json
  COMMAND ${PYTHON} -m ${DESIGN_MODULE}.quant.pack
          --checkpoint-config ${GENERATED_DIR}/checkpoint.json --compiler-build ${COMPILER_BUILD}
          --kernels ${KERNEL_DIR}/kernels.json --output ${GENERATED_DIR}/weights
  DEPENDS ${KERNEL_DIR}/kernels.json ${MODEL_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/quant/pack.py ${CMAKE_CURRENT_SOURCE_DIR}/trace/moe.py
          ${CMAKE_CURRENT_SOURCE_DIR}/permute/layout.py
          ${BUDDY_SOURCE}/frontend/Python/graph/transform/quantization/mxfp8.py
          ${BUDDY_SOURCE}/frontend/Python/graph/transform/layout/matrix.py
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
add_custom_target(omni-weights DEPENDS ${GENERATED_DIR}/weights/placements.json)
set(FORWARD_PASSES
  -buffer-deallocation-simplification -convert-linalg-to-loops -eliminate-empty-tensors
  -llvm-request-c-wrappers -convert-math-to-llvm -convert-math-to-libm -lower-affine
  -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -expand-strided-metadata
  -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts
  -canonicalize -cse -lower-tile-runtime)
set(KERNEL_PASSES
  -eliminate-empty-tensors -convert-elementwise-to-linalg -convert-tensor-to-linalg
  -one-shot-bufferize=bufferize-function-boundaries -convert-linalg-to-tile=quantized-only
  -convert-tile-to-buckyball -lower-buckyball-to-bank-ssa -assign-physical-banks
  -llvm-request-c-wrappers -lower-bank-ssa-to-intrinsics -expand-strided-metadata
  -convert-linalg-to-loops -lower-buckyball -convert-vector-to-scf=target-rank=0
  -lower-affine -canonicalize -cse -convert-scf-to-cf -convert-cf-to-llvm
  -convert-vector-to-llvm -buffer-deallocation-simplification -bufferization-lower-deallocations
  -convert-math-to-llvm -convert-math-to-libm -convert-arith-to-llvm -convert-func-to-llvm
  -finalize-memref-to-llvm -reconcile-unrealized-casts -canonicalize -cse -lower-tile-runtime)
set(OBJECTS)
foreach(STAGE IN LISTS STAGES)
  set(TARGET attention)
  if(STAGE STREQUAL "expert" OR STAGE STREQUAL "output")
    set(TARGET ffn)
  endif()
  compile_graph(${STAGE} "${KERNEL_DIR}/${STAGE}.mlir" ${TARGET} kernel)
  compile_graph(${STAGE}-forward "${KERNEL_DIR}/${STAGE}-forward.mlir" ${TARGET} forward)
  list(APPEND OBJECTS ${STAGE}.o ${STAGE}-forward.o)
endforeach()
file(GLOB EXECUTION_SOURCES CONFIGURE_DEPENDS "${MODEL_SOURCE}/execution/*.cpp")
add_executable(omni-run ${MODEL_SOURCE}/main.cpp ${EXECUTION_SOURCES} ${OBJECTS} ${KERNEL_DIR}/model-parameters.h)
target_include_directories(omni-run PRIVATE ${KERNEL_DIR} ${BUDDY_SOURCE}/frontend/Interfaces)
target_compile_options(omni-run PRIVATE -O2 -march=rv64gc)
target_link_libraries(omni-run PRIVATE smatmul_runtime runtime -static)
add_dependencies(omni-run model-loader omni-weights)
add_custom_command(TARGET omni-run POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${ARTIFACT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:omni-run> ${ARTIFACT_DIR}/omni-run
  COMMAND ${PYTHON} -m stack.models.models.qwen3-omni-30b-a3b-instruct.pack
          --generated-dir ${GENERATED_DIR} --model-dir ${ARTIFACT_DIR}
          --rax-pack ${COMPILER_BUILD}/bin/rax-pack
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
set_property(TARGET omni-run APPEND PROPERTY LINK_DEPENDS
  ${MODEL_SOURCE}/pack.py ${REPO_ROOT}/stack/compiler/package.py ${GENERATED_DIR}/weights/placements.json)

add_library(smatmul_runtime STATIC ${REPO_ROOT}/examples/balls/smatmul/compiler/runtime/fp32.cpp)
target_include_directories(smatmul_runtime PRIVATE ${RUNTIME_PARAMS}
  ${REPO_ROOT}/stack/runtime/include ${REPO_ROOT}/bb-tests/workloads/lib
  ${REPO_ROOT}/examples/balls/smatmul/workloads
  ${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine)
target_compile_options(smatmul_runtime PRIVATE -O2)
