set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

set(
  STAGES
  prefill_attention
  decode_attention
  prefill_router
  decode_router
  prefill_embedding
  decode_embedding
  prefill_norm
  decode_norm
  expert
  expert_2
  expert_4
  expert_8
  expert_prefill
  output
)
set(KERNEL_FILES)
foreach(STAGE IN LISTS STAGES)
  list(
    APPEND KERNEL_FILES
    ${OUTPUT_DIR}/thinker/${STAGE}.mlir
    ${OUTPUT_DIR}/thinker/${STAGE}-forward.mlir
    ${OUTPUT_DIR}/thinker/${STAGE}-task.c
  )
endforeach()

add_custom_command(
  OUTPUT ${KERNEL_FILES}
         ${OUTPUT_DIR}/thinker/kernels.json
         ${OUTPUT_DIR}/thinker/model-parameters.h
  COMMAND ${PYTHON} buddy-pytorch-importer.py
            --repo-root ${REPO_ROOT}
            --checkpoint-config ${MODEL_DIR}/qwen3-omni-30b-a3b-instruct/output/checkpoint.json
            --compiler-build ${COMPILER_BUILD}
            --chip-pb ${CHIP_PB}
            --output ${OUTPUT_DIR}/thinker
            --system-config ${REPO_ROOT}/examples/chips/omni/configs/system.toml
  DEPENDS ${MODEL_DIR}/qwen3-omni-30b-a3b-instruct/output/checkpoint.json
          ${MODEL_DIR}/qwen3-omni-30b-a3b-instruct/attention.py
          ${MODEL_DIR}/qwen3-omni-30b-a3b-instruct/moe.py
          ${MODEL_DIR}/qwen3-omni-30b-a3b-instruct/projections.py
          ${MODEL_DIR}/qwen3-omni-30b-a3b-instruct/weights.py
          ${CMAKE_CURRENT_SOURCE_DIR}/trace/graph.py
          ${CMAKE_CURRENT_SOURCE_DIR}/permute/layout.py
          ${CMAKE_CURRENT_SOURCE_DIR}/trace/moe.py
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${CMAKE_CURRENT_SOURCE_DIR}/configs/compiler-param.toml
          ${REPO_ROOT}/examples/chips/omni/configs/system.toml
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/weights/placements.json
  COMMAND ${PYTHON} pack-weights.py
            --repo-root ${REPO_ROOT}
            --checkpoint-config ${MODEL_DIR}/qwen3-omni-30b-a3b-instruct/output/checkpoint.json
            --compiler-build ${COMPILER_BUILD}
            --kernels ${OUTPUT_DIR}/thinker/kernels.json
            --output ${OUTPUT_DIR}/weights
  DEPENDS ${OUTPUT_DIR}/thinker/kernels.json
          ${MODEL_DIR}/qwen3-omni-30b-a3b-instruct/attention.py
          ${MODEL_DIR}/qwen3-omni-30b-a3b-instruct/moe.py
          ${MODEL_DIR}/qwen3-omni-30b-a3b-instruct/projections.py
          ${MODEL_DIR}/qwen3-omni-30b-a3b-instruct/weights.py
          ${CMAKE_CURRENT_SOURCE_DIR}/pack-weights.py
          ${CMAKE_CURRENT_SOURCE_DIR}/trace/moe.py
          ${CMAKE_CURRENT_SOURCE_DIR}/permute/layout.py
          ${MODEL_DIR}/qwen3-omni-30b-a3b-instruct/weights.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)
add_custom_target(omni-weights DEPENDS ${OUTPUT_DIR}/weights/placements.json)
set(OBJECTS)
foreach(STAGE IN LISTS STAGES)
  set(TARGET attention)
  if(STAGE MATCHES "^expert" OR STAGE STREQUAL "output")
    set(TARGET ffn)
  endif()

  add_custom_command(
    OUTPUT ${STAGE}.o
    BYPRODUCTS ${OUTPUT_DIR}/${STAGE}_linalg.mlir
               ${OUTPUT_DIR}/${STAGE}_tile.mlir
               ${OUTPUT_DIR}/${STAGE}_buckyball.mlir
               ${OUTPUT_DIR}/${STAGE}_bank_ssa.mlir
               ${OUTPUT_DIR}/${STAGE}_llvm.mlir
    COMMAND ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt ${OUTPUT_DIR}/thinker/${STAGE}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
              -o ${OUTPUT_DIR}/${STAGE}_linalg.mlir
    COMMAND ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt ${OUTPUT_DIR}/thinker/${STAGE}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -fuse-pointwise
              -outline-rvv-kernels
              -canonicalize
              -canonicalize
              -cse
              -one-shot-bufferize=bufferize-function-boundaries
              -lower-rvv-kernels
              -buffer-deallocation-pipeline=private-function-dynamic-ownership=false
              -convert-linalg-to-tile=quantized-only
              -o ${OUTPUT_DIR}/${STAGE}_tile.mlir
    COMMAND ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt ${OUTPUT_DIR}/thinker/${STAGE}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -fuse-pointwise
              -outline-rvv-kernels
              -canonicalize
              -canonicalize
              -cse
              -one-shot-bufferize=bufferize-function-boundaries
              -lower-rvv-kernels
              -buffer-deallocation-pipeline=private-function-dynamic-ownership=false
              -convert-linalg-to-tile=quantized-only
              -convert-tile-to-buckyball
              -o ${OUTPUT_DIR}/${STAGE}_buckyball.mlir
    COMMAND ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt ${OUTPUT_DIR}/thinker/${STAGE}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -fuse-pointwise
              -outline-rvv-kernels
              -canonicalize
              -canonicalize
              -cse
              -one-shot-bufferize=bufferize-function-boundaries
              -lower-rvv-kernels
              -buffer-deallocation-pipeline=private-function-dynamic-ownership=false
              -convert-linalg-to-tile=quantized-only
              -convert-tile-to-buckyball
              -lower-buckyball-to-bank-ssa
              -assign-physical-banks
              -o ${OUTPUT_DIR}/${STAGE}_bank_ssa.mlir
    COMMAND ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt ${OUTPUT_DIR}/thinker/${STAGE}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -fuse-pointwise
              -outline-rvv-kernels
              -canonicalize
              -canonicalize
              -cse
              -one-shot-bufferize=bufferize-function-boundaries
              -lower-rvv-kernels
              -buffer-deallocation-pipeline=private-function-dynamic-ownership=false
              -convert-linalg-to-tile=quantized-only
              -convert-tile-to-buckyball
              -lower-buckyball-to-bank-ssa
              -assign-physical-banks
              -llvm-request-c-wrappers
              -lower-bank-ssa-to-intrinsics
              -expand-strided-metadata
              -convert-linalg-to-loops
              -lower-buckyball
              -convert-vector-to-scf=target-rank=0
              -lower-affine
              -canonicalize
              -cse
              -convert-scf-to-cf
              -convert-cf-to-llvm
              -convert-vector-to-llvm
              -buffer-deallocation-simplification
              -bufferization-lower-deallocations
              -convert-math-to-llvm
              -convert-math-to-libm
              -convert-arith-to-llvm
              -convert-func-to-llvm
              -finalize-memref-to-llvm
              -reconcile-unrealized-casts
              -canonicalize
              -cse
              -o ${OUTPUT_DIR}/${STAGE}_llvm.mlir
    COMMAND ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt ${OUTPUT_DIR}/thinker/${STAGE}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -fuse-pointwise
              -outline-rvv-kernels
              -canonicalize
              -canonicalize
              -cse
              -one-shot-bufferize=bufferize-function-boundaries
              -lower-rvv-kernels
              -buffer-deallocation-pipeline=private-function-dynamic-ownership=false
              -convert-linalg-to-tile=quantized-only
              -convert-tile-to-buckyball
              -lower-buckyball-to-bank-ssa
              -assign-physical-banks
              -llvm-request-c-wrappers
              -lower-bank-ssa-to-intrinsics
              -expand-strided-metadata
              -convert-linalg-to-loops
              -lower-buckyball
              -convert-vector-to-scf=target-rank=0
              -lower-affine
              -canonicalize
              -cse
              -convert-scf-to-cf
              -convert-cf-to-llvm
              -convert-vector-to-llvm
              -buffer-deallocation-simplification
              -bufferization-lower-deallocations
              -convert-math-to-llvm
              -convert-math-to-libm
              -convert-arith-to-llvm
              -convert-func-to-llvm
              -finalize-memref-to-llvm
              -reconcile-unrealized-casts
              -canonicalize
              -cse |
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-translate
              --buddy-to-llvmir |
            ${BUDDY_SOURCE}/llvm/build/bin/opt -passes=default<O2> |
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O3 -mattr=+xbuckyball,+m,+D,+c -float-abi=hard -o ${STAGE}.o
    COMMAND ${CMAKE_OBJCOPY}
              --redefine-syms=${THREAD_RUNTIME_DIR}/${TARGET}.rename
              --redefine-sym malloc=workspace_alloc
              --redefine-sym free=workspace_free ${STAGE}.o
    DEPENDS "${OUTPUT_DIR}/thinker/${STAGE}.mlir"
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-translate
            "${BUDDY_SOURCE}/llvm/build/bin/opt"
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-llc
            ${THREAD_RUNTIME_DIR}/${TARGET}.rename
    COMMENT "Building ${STAGE}.o"
    VERBATIM
  )

  add_custom_command(
    OUTPUT ${STAGE}-forward.o
    BYPRODUCTS ${OUTPUT_DIR}/${STAGE}-forward_llvm.mlir
    COMMAND ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt ${OUTPUT_DIR}/thinker/${STAGE}-forward.mlir
              -buffer-deallocation-simplification
              -convert-linalg-to-loops
              -eliminate-empty-tensors
              -llvm-request-c-wrappers
              -convert-math-to-llvm
              -convert-math-to-libm
              -lower-affine
              -convert-scf-to-cf
              -convert-cf-to-llvm
              -convert-arith-to-llvm
              -expand-strided-metadata
              -finalize-memref-to-llvm
              -convert-func-to-llvm
              -reconcile-unrealized-casts
              -canonicalize
              -cse
              -o ${OUTPUT_DIR}/${STAGE}-forward_llvm.mlir
    COMMAND ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt ${OUTPUT_DIR}/thinker/${STAGE}-forward.mlir
              -buffer-deallocation-simplification
              -convert-linalg-to-loops
              -eliminate-empty-tensors
              -llvm-request-c-wrappers
              -convert-math-to-llvm
              -convert-math-to-libm
              -lower-affine
              -convert-scf-to-cf
              -convert-cf-to-llvm
              -convert-arith-to-llvm
              -expand-strided-metadata
              -finalize-memref-to-llvm
              -convert-func-to-llvm
              -reconcile-unrealized-casts
              -canonicalize
              -cse |
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-translate
              --buddy-to-llvmir |
            ${BUDDY_SOURCE}/llvm/build/bin/opt -passes=default<O2> |
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O3 -mattr=+xbuckyball,+m,+D,+c -float-abi=hard -o ${STAGE}-forward.o
    COMMAND ${CMAKE_OBJCOPY}
              --redefine-syms=${THREAD_RUNTIME_DIR}/${TARGET}.rename
              --redefine-sym malloc=workspace_alloc
              --redefine-sym free=workspace_free ${STAGE}-forward.o
    DEPENDS "${OUTPUT_DIR}/thinker/${STAGE}-forward.mlir"
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-translate
            "${BUDDY_SOURCE}/llvm/build/bin/opt"
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-llc
            ${THREAD_RUNTIME_DIR}/${TARGET}.rename
    COMMENT "Building ${STAGE}-forward.o"
    VERBATIM
  )
  list(APPEND OBJECTS ${STAGE}.o ${STAGE}-forward.o ${OUTPUT_DIR}/thinker/${STAGE}-task.c)
endforeach()
include("${CMAKE_CURRENT_SOURCE_DIR}/media/vision/CMakeLists.txt")
include("${CMAKE_CURRENT_SOURCE_DIR}/media/audio/CMakeLists.txt")
include("${CMAKE_CURRENT_SOURCE_DIR}/media/talker/CMakeLists.txt")
include("${CMAKE_CURRENT_SOURCE_DIR}/media/wave/CMakeLists.txt")
file(GLOB EXECUTION_SOURCES CONFIGURE_DEPENDS "${MODEL_DIR}/qwen3-omni-30b-a3b-instruct/execution/*.cpp")
add_executable(omni-run EXCLUDE_FROM_ALL ${MODEL_DIR}/qwen3-omni-30b-a3b-instruct/main.cpp ${EXECUTION_SOURCES}
  ${OBJECTS} ${OUTPUT_DIR}/thinker/model-parameters.h ${OUTPUT_DIR}/vision/vision-parameters.h
  ${OUTPUT_DIR}/audio/audio-parameters.h ${OUTPUT_DIR}/talker/talker-parameters.h
  ${OUTPUT_DIR}/wave/wave-parameters.h
)
set_target_properties(omni-run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
target_include_directories(omni-run PRIVATE ${MODEL_DIR}/qwen3-omni-30b-a3b-instruct/execution
                                            ${OUTPUT_DIR}/thinker
                                            ${OUTPUT_DIR}/vision
                                            ${OUTPUT_DIR}/audio
                                            ${OUTPUT_DIR}/talker
                                            ${OUTPUT_DIR}/wave
                                            ${CMAKE_BINARY_DIR}/isa
                                            ${BUDDY_SOURCE}/frontend/Interfaces
                                            ${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine
)
target_compile_options(omni-run PRIVATE -O2 -march=rv64gc)
target_link_libraries(omni-run PRIVATE runtime
                                       rvv_kernels
                                       mxfp8_pack
                                       rvv_quant_backend
                                       -static
)
add_dependencies(
  omni-run
  model-loader
  omni-weights
  omni-vision-weights
  omni-audio-weights
  omni-talker-weights
  omni-wave-weights
)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/qwen3-omni.rax
  BYPRODUCTS ${OUTPUT_DIR}/layout.json
             ${OUTPUT_DIR}/qwen3-omni.rhal.mlir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
  COMMAND ${PYTHON} pack.py
            --repo-root ${REPO_ROOT}
            --output-dir ${OUTPUT_DIR}
            --rax-pack ${COMPILER_BUILD}/bin/rax-pack
  DEPENDS omni-run
          ${CMAKE_CURRENT_SOURCE_DIR}/pack.py
          ${OUTPUT_DIR}/weights/placements.json
          ${OUTPUT_DIR}/weights/vision-placement.json
          ${OUTPUT_DIR}/weights/audio-placement.json
          ${OUTPUT_DIR}/weights/talker-placement.json
          ${OUTPUT_DIR}/weights/wave-placement.json
          ${MODEL_DIR}/qwen3-omni-30b-a3b-instruct/output/checkpoint.json
          ${KERNEL_FILES}
          ${OUTPUT_DIR}/thinker/kernels.json
          ${OUTPUT_DIR}/thinker/model-parameters.h
          ${COMPILER_BUILD}/bin/rax-pack
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging qwen3-omni-30b-a3b-instruct model"
  VERBATIM
)
add_custom_target(qwen3-omni-30b-a3b-instruct-model-build DEPENDS ${OUTPUT_DIR}/qwen3-omni.rax)

add_dependencies(qwen3-omni-30b-a3b-instruct-model-build qwen3-omni-30b-a3b-instruct-prepare)
