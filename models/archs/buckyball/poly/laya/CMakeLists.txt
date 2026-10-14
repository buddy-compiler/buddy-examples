set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

file(GLOB_RECURSE DESIGN_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/trace/*.json"
)
list(APPEND DESIGN_FILES ${REPO_ROOT}/stack/compiler/quant/mxfp8_embedding.py)
set(STAGES embedding)
foreach(LAYER RANGE 0 27)
  list(APPEND STAGES attention_${LAYER} ffn_${LAYER})
endforeach()
list(
  APPEND STAGES
  typed
  head_attention_0
  head_ffn_0
  head_attention_1
  head_ffn_1
  scorer
  action
)
set(GRAPHS)
foreach(STAGE IN LISTS STAGES)
  list(APPEND GRAPHS ${OUTPUT_DIR}/${STAGE}.mlir ${OUTPUT_DIR}/${STAGE}-forward.mlir ${OUTPUT_DIR}/${STAGE}-task.c)
endforeach()
add_custom_command(
  OUTPUT ${GRAPHS}
         ${OUTPUT_DIR}/stages.h
         ${OUTPUT_DIR}/model.json
  COMMAND ${PYTHON} buddy-pytorch-importer.py
            --repo-root ${REPO_ROOT}
            --checkpoint "convaiinnovations/laya"
            --revision "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
            --sequence-length 128
            --options 16
            --output-dir ${OUTPUT_DIR}
            --compiler-build ${COMPILER_BUILD}
            --chip-pb ${CHIP_PB}
  DEPENDS ${MODEL_DIR}/laya/configs/importer-param.py
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${MODEL_DIR}/laya/model.py
          ${MODEL_DIR}/laya/inputs.py
          ${DESIGN_FILES}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)
set(OBJECTS)
foreach(STAGE IN LISTS STAGES)
  if(STAGE MATCHES "ffn" OR STAGE STREQUAL "scorer" OR STAGE STREQUAL "action")
    set(TARGET ffn)
  else()
    set(TARGET attention)
  endif()

  add_custom_command(
    OUTPUT ${STAGE}.o
    BYPRODUCTS ${OUTPUT_DIR}/${STAGE}_linalg.mlir
               ${OUTPUT_DIR}/${STAGE}_tile.mlir
               ${OUTPUT_DIR}/${STAGE}_buckyball.mlir
               ${OUTPUT_DIR}/${STAGE}_bank_ssa.mlir
               ${OUTPUT_DIR}/${STAGE}_llvm.mlir
    COMMAND ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt ${OUTPUT_DIR}/${STAGE}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
              -o ${OUTPUT_DIR}/${STAGE}_linalg.mlir
    COMMAND ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt ${OUTPUT_DIR}/${STAGE}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -fuse-pointwise
              -outline-rvv-matmul
              -outline-rvv-kernels
              -fold-unit-transpose
              -canonicalize
              -canonicalize
              -cse
              -one-shot-bufferize=bufferize-function-boundaries
              -lower-rvv-kernels
              -lower-rvv-pointwise
              -lower-gather-rows
              -fuse-norm-quant-window
              -buffer-deallocation-pipeline=private-function-dynamic-ownership=false
              -convert-linalg-to-tile=quantized-only
              -o ${OUTPUT_DIR}/${STAGE}_tile.mlir
    COMMAND ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt ${OUTPUT_DIR}/${STAGE}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -fuse-pointwise
              -outline-rvv-matmul
              -outline-rvv-kernels
              -fold-unit-transpose
              -canonicalize
              -canonicalize
              -cse
              -one-shot-bufferize=bufferize-function-boundaries
              -lower-rvv-kernels
              -lower-rvv-pointwise
              -lower-gather-rows
              -fuse-norm-quant-window
              -buffer-deallocation-pipeline=private-function-dynamic-ownership=false
              -convert-linalg-to-tile=quantized-only
              -convert-tile-to-buckyball
              -o ${OUTPUT_DIR}/${STAGE}_buckyball.mlir
    COMMAND ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt ${OUTPUT_DIR}/${STAGE}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -fuse-pointwise
              -outline-rvv-matmul
              -outline-rvv-kernels
              -fold-unit-transpose
              -canonicalize
              -canonicalize
              -cse
              -one-shot-bufferize=bufferize-function-boundaries
              -lower-rvv-kernels
              -lower-rvv-pointwise
              -lower-gather-rows
              -fuse-norm-quant-window
              -buffer-deallocation-pipeline=private-function-dynamic-ownership=false
              -convert-linalg-to-tile=quantized-only
              -convert-tile-to-buckyball
              -lower-buckyball-to-bank-ssa
              -assign-physical-banks
              -o ${OUTPUT_DIR}/${STAGE}_bank_ssa.mlir
    COMMAND ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt ${OUTPUT_DIR}/${STAGE}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -fuse-pointwise
              -outline-rvv-matmul
              -outline-rvv-kernels
              -fold-unit-transpose
              -canonicalize
              -canonicalize
              -cse
              -one-shot-bufferize=bufferize-function-boundaries
              -lower-rvv-kernels
              -lower-rvv-pointwise
              -lower-gather-rows
              -fuse-norm-quant-window
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
              -lower-ant-runtime
              -verify-npu-compute
              -o ${OUTPUT_DIR}/${STAGE}_llvm.mlir
    COMMAND ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt ${OUTPUT_DIR}/${STAGE}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -fuse-pointwise
              -outline-rvv-matmul
              -outline-rvv-kernels
              -fold-unit-transpose
              -canonicalize
              -canonicalize
              -cse
              -one-shot-bufferize=bufferize-function-boundaries
              -lower-rvv-kernels
              -lower-rvv-pointwise
              -lower-gather-rows
              -fuse-norm-quant-window
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
              -lower-ant-runtime
              -verify-npu-compute |
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-translate
              --buddy-to-llvmir |
            ${BUDDY_SOURCE}/llvm/build/bin/opt "-passes=function(sroa,early-cse,instcombine,simplifycfg,adce)" |
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o ${STAGE}.o
    COMMAND ${CMAKE_OBJCOPY}
              --redefine-syms=${THREAD_RUNTIME_DIR}/${TARGET}.rename
              --redefine-sym malloc=workspace_alloc
              --redefine-sym free=workspace_free ${STAGE}.o
    DEPENDS "${OUTPUT_DIR}/${STAGE}.mlir"
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
    COMMAND ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt ${OUTPUT_DIR}/${STAGE}-forward.mlir
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
    COMMAND ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt ${OUTPUT_DIR}/${STAGE}-forward.mlir
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
            ${BUDDY_SOURCE}/llvm/build/bin/opt "-passes=function(sroa,early-cse,instcombine,simplifycfg,adce)" |
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o ${STAGE}-forward.o
    COMMAND ${CMAKE_OBJCOPY}
              --redefine-syms=${THREAD_RUNTIME_DIR}/${TARGET}.rename
              --redefine-sym malloc=workspace_alloc
              --redefine-sym free=workspace_free ${STAGE}-forward.o
    DEPENDS "${OUTPUT_DIR}/${STAGE}-forward.mlir"
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-translate
            "${BUDDY_SOURCE}/llvm/build/bin/opt"
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-llc
            ${THREAD_RUNTIME_DIR}/${TARGET}.rename
    COMMENT "Building ${STAGE}-forward.o"
    VERBATIM
  )
  list(APPEND OBJECTS ${STAGE}.o ${STAGE}-forward.o ${OUTPUT_DIR}/${STAGE}-task.c)
endforeach()
add_library(laya-buckyball-poly-model STATIC EXCLUDE_FROM_ALL ${OBJECTS})
set_target_properties(laya-buckyball-poly-model PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
set_target_properties(laya-buckyball-poly-model PROPERTIES LINKER_LANGUAGE CXX)

add_executable(laya-run EXCLUDE_FROM_ALL ${MODEL_DIR}/laya/main.cpp ${MODEL_DIR}/laya/execution.cpp
  ${MODEL_DIR}/laya/decision.cpp ${MODEL_DIR}/laya/model.h
)
set_target_properties(laya-run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
add_dependencies(laya-run model-loader)
target_include_directories(laya-run PRIVATE ${ISA_DIR}/chip
                                            ${OUTPUT_DIR}
                                            ${BUDDY_SOURCE}/frontend/Interfaces
)
target_link_libraries(laya-run PRIVATE -static
                                       laya-buckyball-poly-model
                                       runtime_ant_tasks
                                       rvv_kernels_ant
                                       thread_runtime_ant
)
target_compile_options(laya-run PRIVATE -O2)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/laya.rax
  BYPRODUCTS ${OUTPUT_DIR}/layout.json
             ${OUTPUT_DIR}/laya.rhal.mlir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
  COMMAND ${PYTHON} pack.py
            --repo-root ${REPO_ROOT}
            --output-dir ${OUTPUT_DIR}
            --run-config ${CMAKE_CURRENT_SOURCE_DIR}/configs/running-param.toml
            --rax-pack ${COMPILER_BUILD}/bin/rax-pack
  DEPENDS laya-run
          ${CMAKE_CURRENT_SOURCE_DIR}/pack.py
          ${MODEL_DIR}/laya/inputs.py
          ${CMAKE_CURRENT_SOURCE_DIR}/configs/running-param.toml
          ${CMAKE_CURRENT_SOURCE_DIR}/trace/requests.json
          ${GRAPHS}
          ${OUTPUT_DIR}/stages.h
          ${OUTPUT_DIR}/model.json
          ${COMPILER_BUILD}/bin/rax-pack
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging laya model"
  VERBATIM
)
add_custom_target(laya-model-build DEPENDS ${OUTPUT_DIR}/laya.rax)
