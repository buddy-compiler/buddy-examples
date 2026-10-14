set(VISION_STAGES)
foreach(COUNT 4 16 64 256)
  foreach(
    KIND
    patch
    attention
    mlp
    merge
    deepstack
  )
    list(APPEND VISION_STAGES vision_${KIND}_${COUNT})
  endforeach()
endforeach()
set(VISION_FILES)
foreach(STAGE IN LISTS VISION_STAGES)
  list(
    APPEND VISION_FILES
    ${OUTPUT_DIR}/vision/${STAGE}.mlir
    ${OUTPUT_DIR}/vision/${STAGE}-forward.mlir
    ${OUTPUT_DIR}/vision/${STAGE}-task.c
  )
endforeach()
add_custom_command(
  OUTPUT ${VISION_FILES}
         ${OUTPUT_DIR}/vision/kernels.json
         ${OUTPUT_DIR}/vision/vision-parameters.h
  COMMAND ${PYTHON} buddy-pytorch-vision-importer.py
            --repo-root ${REPO_ROOT}
            --checkpoint-config ${MODEL_DIR}/qwen3-omni-30b-a3b-instruct/output/checkpoint.json
            --compiler-build ${COMPILER_BUILD}
            --chip-pb ${CHIP_PB}
            --output ${OUTPUT_DIR}/vision
  DEPENDS ${MODEL_DIR}/qwen3-omni-30b-a3b-instruct/output/checkpoint.json
          ${REPO_ROOT}/stack/models/models/qwen3-omni-30b-a3b-instruct/vision.py
          ${REPO_ROOT}/stack/models/models/qwen3-omni-30b-a3b-instruct/weights.py
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-vision-importer.py
          ${CMAKE_CURRENT_SOURCE_DIR}/trace/graph.py
          ${CMAKE_CURRENT_SOURCE_DIR}/permute/layout.py
          ${CMAKE_CURRENT_SOURCE_DIR}/configs/compiler-param.toml
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/weights/vision-placement.json
  COMMAND ${PYTHON} pack-vision-weights.py
            --repo-root ${REPO_ROOT}
            --checkpoint-config ${MODEL_DIR}/qwen3-omni-30b-a3b-instruct/output/checkpoint.json
            --compiler-build ${COMPILER_BUILD}
            --kernels ${OUTPUT_DIR}/vision/kernels.json
            --system-config ${REPO_ROOT}/examples/chips/omni/configs/system.toml
            --output ${OUTPUT_DIR}/weights
  DEPENDS ${OUTPUT_DIR}/vision/kernels.json
          ${REPO_ROOT}/stack/models/models/qwen3-omni-30b-a3b-instruct/vision.py
          ${REPO_ROOT}/stack/models/models/qwen3-omni-30b-a3b-instruct/weights.py
          ${CMAKE_CURRENT_SOURCE_DIR}/pack-vision-weights.py
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-vision-importer.py
          ${CMAKE_CURRENT_SOURCE_DIR}/permute/layout.py
          ${REPO_ROOT}/stack/models/models/qwen3-omni-30b-a3b-instruct/weights.py
          ${REPO_ROOT}/examples/chips/omni/configs/system.toml
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)
add_custom_target(omni-vision-weights DEPENDS ${OUTPUT_DIR}/weights/vision-placement.json)
foreach(STAGE IN LISTS VISION_STAGES)
  set(TARGET ffn)
  if(STAGE MATCHES "^vision_(patch|attention)_")
    set(TARGET attention)
  endif()

  add_custom_command(
    OUTPUT ${STAGE}.o
    BYPRODUCTS ${OUTPUT_DIR}/${STAGE}_linalg.mlir
               ${OUTPUT_DIR}/${STAGE}_tile.mlir
               ${OUTPUT_DIR}/${STAGE}_buckyball.mlir
               ${OUTPUT_DIR}/${STAGE}_bank_ssa.mlir
               ${OUTPUT_DIR}/${STAGE}_llvm.mlir
    COMMAND ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt ${OUTPUT_DIR}/vision/${STAGE}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
              -o ${OUTPUT_DIR}/${STAGE}_linalg.mlir
    COMMAND ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt ${OUTPUT_DIR}/vision/${STAGE}.mlir
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
    COMMAND ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt ${OUTPUT_DIR}/vision/${STAGE}.mlir
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
    COMMAND ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt ${OUTPUT_DIR}/vision/${STAGE}.mlir
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
    COMMAND ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt ${OUTPUT_DIR}/vision/${STAGE}.mlir
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
    COMMAND ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt ${OUTPUT_DIR}/vision/${STAGE}.mlir
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
    DEPENDS "${OUTPUT_DIR}/vision/${STAGE}.mlir"
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
    COMMAND ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt ${OUTPUT_DIR}/vision/${STAGE}-forward.mlir
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
    COMMAND ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt ${OUTPUT_DIR}/vision/${STAGE}-forward.mlir
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
    DEPENDS "${OUTPUT_DIR}/vision/${STAGE}-forward.mlir"
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-translate
            "${BUDDY_SOURCE}/llvm/build/bin/opt"
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-llc
            ${THREAD_RUNTIME_DIR}/${TARGET}.rename
    COMMENT "Building ${STAGE}-forward.o"
    VERBATIM
  )
  list(APPEND OBJECTS ${STAGE}.o ${STAGE}-forward.o ${OUTPUT_DIR}/vision/${STAGE}-task.c)
endforeach()
