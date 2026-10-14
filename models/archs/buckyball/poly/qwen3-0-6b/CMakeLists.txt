set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

file(STRINGS "${ISA_DIR}/chip/topology.h" TILE_COUNT_LINE REGEX "^#define BB_COMPUTE_TILES ")
string(REGEX REPLACE "^#define BB_COMPUTE_TILES ([0-9]+)$" "\\1" EXECUTION_TILES "${TILE_COUNT_LINE}")
if(NOT EXECUTION_TILES MATCHES "^[0-9]+$" OR EXECUTION_TILES LESS 2)
  message(FATAL_ERROR "Qwen requires at least two compute tiles")
endif()
math(EXPR UNPAIRED_TILE "${EXECUTION_TILES} % 2")
if(UNPAIRED_TILE)
  message(FATAL_ERROR "Qwen requires paired compute tiles")
endif()
math(EXPR PARTS "${EXECUTION_TILES} / 2")
execute_process(
  COMMAND
    ${PYTHON} -c
    "import json,sys; from huggingface_hub import hf_hub_download; c=json.load(open(hf_hub_download(sys.argv[1], 'config.json'))); print(';'.join(str(c[k]) for k in ('hidden_size','intermediate_size')))"
    "Qwen/Qwen3-0.6B"
  OUTPUT_VARIABLE MODEL_DIMENSIONS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
list(GET MODEL_DIMENSIONS 0 HIDDEN_SIZE)
list(GET MODEL_DIMENSIONS 1 INTERMEDIATE_SIZE)
set(
  STAGES
  prefill_embedding
  decode_embedding
  prefill_attention_body
  prefill_attention_projection
  decode_attention_body
  decode_attention_projection
)
math(EXPR FFN_INTERMEDIATE "${INTERMEDIATE_SIZE} / ${PARTS}")
math(EXPR FFN_EXPAND_WIDTH "((${FFN_INTERMEDIATE} / 2 / 32 + 2) / 3) * 32")
math(EXPR FFN_EXPAND_LAST "${FFN_INTERMEDIATE} / 2 - 2 * ${FFN_EXPAND_WIDTH}")
math(EXPR FFN_DOWN_WIDTH "((${HIDDEN_SIZE} / 2 / 16 + 2) / 3) * 16")
math(EXPR FFN_DOWN_LAST "${HIDDEN_SIZE} / 2 - 2 * ${FFN_DOWN_WIDTH}")
foreach(PHASE prefill decode)
  foreach(WIDTH ${FFN_EXPAND_WIDTH} ${FFN_EXPAND_LAST})
    list(APPEND STAGES ${PHASE}_ffn_expand_${WIDTH})
  endforeach()
  foreach(WIDTH ${FFN_DOWN_WIDTH} ${FFN_DOWN_LAST})
    list(APPEND STAGES ${PHASE}_ffn_down_${WIDTH})
  endforeach()
endforeach()
list(REMOVE_DUPLICATES STAGES)
list(APPEND STAGES output)
foreach(TARGET attention ffn)
  if(NOT "${TARGET}" IN_LIST TARGETS)
    message(FATAL_ERROR "Qwen requires the ${TARGET} core type")
  endif()
endforeach()
set(QWEN_MLIR)
foreach(STAGE IN LISTS STAGES)
  list(APPEND QWEN_MLIR ${OUTPUT_DIR}/${STAGE}.mlir ${OUTPUT_DIR}/${STAGE}-forward.mlir ${OUTPUT_DIR}/${STAGE}-task.c)
endforeach()
file(GLOB_RECURSE DESIGN_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/quant/*.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/trace/*.py" "${CMAKE_CURRENT_SOURCE_DIR}/permute/*.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/trace/*.toml"
)
add_custom_command(
  OUTPUT ${QWEN_MLIR}
         ${OUTPUT_DIR}/model.json
         ${OUTPUT_DIR}/qwen-parameters.h
  BYPRODUCTS ${OUTPUT_DIR}/tokenizer/config.json
             ${OUTPUT_DIR}/tokenizer/tokenizer.json
             ${OUTPUT_DIR}/tokenizer/tokenizer_config.json
  COMMAND ${PYTHON} buddy-pytorch-importer.py
            --repo-root ${REPO_ROOT}
            --output-dir ${OUTPUT_DIR}
            --compiler-build ${COMPILER_BUILD}
            --chip-pb ${CHIP_PB}
            --prefill-len 128
            --max-cache-len 256
            --parts ${PARTS}
            --model Qwen/Qwen3-0.6B
  DEPENDS ${CHIP_PB}
          ${CMAKE_CURRENT_SOURCE_DIR}/config.py
          ${DESIGN_FILES}
          ${MODEL_DIR}/qwen3-0-6b/prepare.py
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${CMAKE_CURRENT_SOURCE_DIR}/configs/importer-param.py
          ${CMAKE_CURRENT_SOURCE_DIR}/trace/stages.py
          ${CMAKE_CURRENT_SOURCE_DIR}/trace/ffn.py
          ${CMAKE_CURRENT_SOURCE_DIR}/trace/output.py
          ${CMAKE_CURRENT_SOURCE_DIR}/trace/attention.py
          ${CMAKE_CURRENT_SOURCE_DIR}/quant/parameters.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)

set(OBJECTS)
foreach(STAGE IN LISTS STAGES)
  if(STAGE MATCHES "_ffn_" OR STAGE STREQUAL "output")
    set(TARGET ffn)
  else()
    set(TARGET attention)
  endif()

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
              -optimize-allocation-liveness
              -convert-linalg-to-tile=quantized-only
              -o ${OUTPUT_DIR}/${STAGE}_tile.mlir
    COMMAND ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt ${OUTPUT_DIR}/${STAGE}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
            ${COMPILER_BUILD_${TARGET}}/bin/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -fuse-pointwise
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
              -optimize-allocation-liveness
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
              -optimize-allocation-liveness
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
              -optimize-allocation-liveness
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
              -optimize-allocation-liveness
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
  list(APPEND OBJECTS ${STAGE}.o ${STAGE}-forward.o ${OUTPUT_DIR}/${STAGE}-task.c)
endforeach()
add_library(qwen3-0-6b-buckyball-poly-model STATIC EXCLUDE_FROM_ALL ${OBJECTS})
set_target_properties(qwen3-0-6b-buckyball-poly-model PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
set_target_properties(qwen3-0-6b-buckyball-poly-model PROPERTIES LINKER_LANGUAGE CXX)

add_executable(qwen3-0-6b-run EXCLUDE_FROM_ALL ${MODEL_DIR}/qwen3-0-6b/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/execution.cpp ${CMAKE_CURRENT_SOURCE_DIR}/step.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/loader.cpp ${MODEL_DIR}/qwen3-0-6b/generation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ffn.cpp ${CMAKE_CURRENT_SOURCE_DIR}/attention.cpp
)
set_target_properties(qwen3-0-6b-run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}" OUTPUT_NAME qwen-run)
target_compile_options(qwen3-0-6b-run PRIVATE -O2)
add_dependencies(qwen3-0-6b-run model-loader)
target_include_directories(qwen3-0-6b-run PRIVATE ${MODEL_DIR}/qwen3-0-6b
                                                  ${ISA_DIR}/chip
                                                  ${OUTPUT_DIR}
                                                  ${BUDDY_SOURCE}/frontend/Interfaces
)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/ffn.cpp PROPERTIES INCLUDE_DIRECTORIES "${ISA_DIR}/ffn")
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/attention.cpp
  PROPERTIES INCLUDE_DIRECTORIES "${ISA_DIR}/attention"
)
target_link_libraries(qwen3-0-6b-run PRIVATE -static
                                             qwen3-0-6b-buckyball-poly-model
                                             runtime_ant_tasks
                                             rvv_kernels_ant
                                             thread_runtime_ant
)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/qwen.rax
  BYPRODUCTS ${OUTPUT_DIR}/layout.json
             ${OUTPUT_DIR}/qwen.rhal.mlir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
  COMMAND ${PYTHON} pack.py
            --repo-root ${REPO_ROOT}
            --output-dir ${OUTPUT_DIR}
            --run-config ${CMAKE_CURRENT_SOURCE_DIR}/configs/running-param.toml
            --backend buckyball
            --rax-pack ${COMPILER_BUILD}/bin/rax-pack
  DEPENDS qwen3-0-6b-run
          ${CMAKE_CURRENT_SOURCE_DIR}/pack.py
          ${CMAKE_CURRENT_SOURCE_DIR}/loader.py
          ${CMAKE_CURRENT_SOURCE_DIR}/configs/running-param.toml
          ${OUTPUT_DIR}/model.json
          ${QWEN_MLIR}
          ${OUTPUT_DIR}/qwen-parameters.h
          ${REPO_ROOT}/examples/models/poly/qwen3-0-6b/pack.py
          ${COMPILER_BUILD}/bin/rax-pack
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging qwen3-0-6b model"
  VERBATIM
)
add_custom_target(qwen3-0-6b-model-build DEPENDS ${OUTPUT_DIR}/qwen.rax)
