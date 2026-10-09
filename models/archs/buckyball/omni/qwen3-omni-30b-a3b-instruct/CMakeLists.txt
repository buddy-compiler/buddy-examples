set(DESIGN_MODULE examples.models.omni.qwen3-omni-30b-a3b-instruct)
set(MODEL_SOURCE "${REPO_ROOT}/stack/models/models/qwen3-omni-30b-a3b-instruct")
set(MODEL_CONFIG_ARGS)
if(DEFINED CHECKPOINT)
  list(APPEND MODEL_CONFIG_ARGS --checkpoint "${CHECKPOINT}")
endif()
if(DEFINED REVISION)
  list(APPEND MODEL_CONFIG_ARGS --revision "${REVISION}")
endif()
set(MODEL_CONFIG_FILES ${MODEL_SOURCE}/configs/model.toml ${MODEL_SOURCE}/configs/importer-param.py)
set(SYSTEM_CONFIG "${REPO_ROOT}/examples/chips/omni/configs/system.toml")
set(KERNEL_DIR "${GENERATED_DIR}/thinker")
set(LLVM_OPT_LEVEL 3)
set(LLVM_PASSES "default<O2>")
set(MATTR "+xbuckyball,+m,+D,+c")
set(STAGES prefill_attention decode_attention prefill_router decode_router
  prefill_embedding decode_embedding prefill_norm decode_norm
  expert expert_2 expert_4 expert_8 expert_prefill output)
set(KERNEL_FILES)
foreach(STAGE IN LISTS STAGES)
  list(APPEND KERNEL_FILES ${KERNEL_DIR}/${STAGE}.mlir ${KERNEL_DIR}/${STAGE}-forward.mlir)
endforeach()
set(GRAPH_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/trace/graph.py
  ${BUDDY_SOURCE}/frontend/Python/ops/tosa.py
  ${COMPILER_BUILD}/python_packages/buddy/compiler/ops/tosa.py
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/layout/provenance.py
  ${COMPILER_BUILD}/python_packages/buddy/compiler/graph/transform/layout/provenance.py
  ${CMAKE_CURRENT_SOURCE_DIR}/permute/layout.py
  ${REPO_ROOT}/stack/compiler/quant/mxfp8_graph.py
  ${REPO_ROOT}/stack/compiler/quant/mxfp8_embedding.py
  ${REPO_ROOT}/stack/compiler/quant/mxfp8_lowering.py
  ${REPO_ROOT}/examples/balls/mxmm/compiler/python/fp32.py
  ${REPO_ROOT}/examples/balls/mxmm/compiler/python/layout.py
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/quantization/mxfp8.py
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/layout/matrix.py)
set(PACK_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/permute/layout.py
  ${MODEL_SOURCE}/weights.py
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/quantization/mxfp8.py
  ${BUDDY_SOURCE}/frontend/Python/graph/transform/layout/matrix.py)
set(MODEL_SOURCES ${MODEL_SOURCE}/attention.py ${MODEL_SOURCE}/moe.py ${MODEL_SOURCE}/projections.py ${MODEL_SOURCE}/weights.py)
add_custom_command(OUTPUT ${GENERATED_DIR}/checkpoint.json
  COMMAND ${PYTHON} -m stack.models.models.qwen3-omni-30b-a3b-instruct.prepare --output-dir ${GENERATED_DIR} ${MODEL_CONFIG_ARGS}
  DEPENDS ${MODEL_CONFIG_FILES} ${MODEL_SOURCE}/prepare.py WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
add_custom_command(OUTPUT ${KERNEL_FILES} ${KERNEL_DIR}/kernels.json ${KERNEL_DIR}/model-parameters.h
  COMMAND ${PYTHON} -m ${DESIGN_MODULE}.trace.kernels
          --checkpoint-config ${GENERATED_DIR}/checkpoint.json
          --compiler-build ${COMPILER_BUILD} --output ${KERNEL_DIR} --system-config ${SYSTEM_CONFIG}
  DEPENDS ${GENERATED_DIR}/checkpoint.json ${MODEL_SOURCES} ${GRAPH_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/trace/moe.py ${CMAKE_CURRENT_SOURCE_DIR}/trace/kernels.py
          ${CMAKE_CURRENT_SOURCE_DIR}/configs/compiler-param.toml ${SYSTEM_CONFIG}
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
add_custom_command(OUTPUT ${GENERATED_DIR}/weights/placements.json
  COMMAND ${PYTHON} -m ${DESIGN_MODULE}.quant.pack
          --checkpoint-config ${GENERATED_DIR}/checkpoint.json --compiler-build ${COMPILER_BUILD}
          --kernels ${KERNEL_DIR}/kernels.json --output ${GENERATED_DIR}/weights
  DEPENDS ${KERNEL_DIR}/kernels.json ${MODEL_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/quant/pack.py ${CMAKE_CURRENT_SOURCE_DIR}/trace/moe.py
          ${PACK_SOURCES}
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
  -fuse-pointwise
  -outline-rvv-kernels -canonicalize
  -canonicalize -cse
  -one-shot-bufferize=bufferize-function-boundaries -lower-rvv-kernels
  -buffer-deallocation-pipeline=private-function-dynamic-ownership=false
  -convert-linalg-to-tile=quantized-only
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
  if(STAGE MATCHES "^expert" OR STAGE STREQUAL "output")
    set(TARGET ffn)
  endif()
  compile_graph(${STAGE} "${KERNEL_DIR}/${STAGE}.mlir" ${TARGET} kernel)
  compile_graph(${STAGE}-forward "${KERNEL_DIR}/${STAGE}-forward.mlir" ${TARGET} forward)
  list(APPEND OBJECTS ${STAGE}.o ${STAGE}-forward.o)
endforeach()
include("${CMAKE_CURRENT_SOURCE_DIR}/media/vision/CMakeLists.txt")
include("${CMAKE_CURRENT_SOURCE_DIR}/media/audio/CMakeLists.txt")
include("${CMAKE_CURRENT_SOURCE_DIR}/media/talker/CMakeLists.txt")
include("${CMAKE_CURRENT_SOURCE_DIR}/media/wave/CMakeLists.txt")
file(GLOB EXECUTION_SOURCES CONFIGURE_DEPENDS "${MODEL_SOURCE}/execution/*.cpp")
add_executable(omni-run ${MODEL_SOURCE}/main.cpp ${EXECUTION_SOURCES} ${OBJECTS} ${KERNEL_DIR}/model-parameters.h ${VISION_DIR}/vision-parameters.h ${AUDIO_DIR}/audio-parameters.h ${TALKER_DIR}/talker-parameters.h ${WAVE_DIR}/wave-parameters.h)
target_include_directories(omni-run PRIVATE ${MODEL_SOURCE}/execution ${KERNEL_DIR} ${VISION_DIR} ${AUDIO_DIR} ${TALKER_DIR} ${WAVE_DIR} ${CMAKE_BINARY_DIR}/isa ${BUDDY_SOURCE}/frontend/Interfaces ${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine)
target_compile_options(omni-run PRIVATE -O2 -march=rv64gc)
add_subdirectory("${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/rvv" rvv)
add_subdirectory("${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/mxfp8" mxfp8)
target_link_libraries(omni-run PRIVATE runtime rvv_kernels mxfp8_pack rvv_quant_backend -static)
add_dependencies(omni-run model-loader omni-weights omni-vision-weights omni-audio-weights omni-talker-weights omni-wave-weights)
add_custom_command(TARGET omni-run POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${ARTIFACT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:omni-run> ${ARTIFACT_DIR}/omni-run
  COMMAND ${PYTHON} -m stack.models.models.qwen3-omni-30b-a3b-instruct.pack
          --generated-dir ${GENERATED_DIR} --model-dir ${ARTIFACT_DIR}
          --rax-pack ${COMPILER_BUILD}/bin/rax-pack
  WORKING_DIRECTORY ${REPO_ROOT} VERBATIM)
set_property(TARGET omni-run APPEND PROPERTY LINK_DEPENDS
  ${MODEL_SOURCE}/pack.py ${REPO_ROOT}/stack/compiler/package.py ${GENERATED_DIR}/weights/placements.json ${GENERATED_DIR}/weights/vision-placement.json ${GENERATED_DIR}/weights/audio-placement.json ${GENERATED_DIR}/weights/talker-placement.json ${GENERATED_DIR}/weights/wave-placement.json)
