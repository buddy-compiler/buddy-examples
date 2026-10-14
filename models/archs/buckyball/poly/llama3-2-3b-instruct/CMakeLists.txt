set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/configs/model.toml")
execute_process(
  COMMAND
    ${PYTHON} -c
    "import sys,tomllib; p=tomllib.load(open(sys.argv[1], 'rb'))['parameters']; print(p['prefill_length']); print(p['cache_length'])"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/model.toml"
  OUTPUT_VARIABLE DESIGN_CAPACITIES
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
string(REPLACE "\n" ";" DESIGN_CAPACITIES "${DESIGN_CAPACITIES}")
list(GET DESIGN_CAPACITIES 0 PREFILL_LENGTH)
list(GET DESIGN_CAPACITIES 1 CACHE_LENGTH)
file(STRINGS "${ISA_DIR}/chip/topology.h" TILE_COUNT_LINE REGEX "^#define BB_COMPUTE_TILES ")
string(REGEX REPLACE "^#define BB_COMPUTE_TILES ([0-9]+)$" "\\1" PARTS "${TILE_COUNT_LINE}")
execute_process(
  COMMAND
    ${PYTHON} -c
    "import json,sys; from huggingface_hub import hf_hub_download; print(json.load(open(hf_hub_download(sys.argv[1], 'config.json')))['num_hidden_layers'])"
    "meta-llama/Llama-3.2-3B-Instruct"
  OUTPUT_VARIABLE LAYERS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
math(EXPR LAST_LAYER "${LAYERS} - 1")
math(EXPR LAST_RANK "${PARTS} - 1")
set(STAGES)
foreach(PHASE prefill decode)
  list(APPEND STAGES ${PHASE}_embedding)
  foreach(RANK RANGE 0 ${LAST_RANK})
    foreach(LAYER RANGE 0 ${LAST_LAYER})
      list(APPEND STAGES ${PHASE}_attention_${LAYER}_rank_${RANK} ${PHASE}_ffn_${LAYER}_rank_${RANK})
    endforeach()
  endforeach()
endforeach()
list(APPEND STAGES output)
foreach(TARGET attention ffn)
  if(NOT "${TARGET}" IN_LIST TARGETS)
    message(FATAL_ERROR "Llama requires the ${TARGET} core type")
  endif()
endforeach()
set(LLAMA_MLIR)
foreach(STAGE IN LISTS STAGES)
  list(APPEND LLAMA_MLIR ${OUTPUT_DIR}/${STAGE}.mlir ${OUTPUT_DIR}/${STAGE}-forward.mlir ${OUTPUT_DIR}/${STAGE}-task.c)
endforeach()
file(GLOB_RECURSE DESIGN_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/quant/*.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/trace/*.py" "${CMAKE_CURRENT_SOURCE_DIR}/trace/*.toml"
  "${CMAKE_CURRENT_SOURCE_DIR}/permute/*.py" "${CMAKE_CURRENT_SOURCE_DIR}/loader/*.py"
)
add_custom_command(
  OUTPUT ${LLAMA_MLIR}
         ${OUTPUT_DIR}/model.json
         ${OUTPUT_DIR}/llama-parameters.h
         ${OUTPUT_DIR}/tokenizer/generation_config.json
  BYPRODUCTS ${OUTPUT_DIR}/tokenizer/config.json
             ${OUTPUT_DIR}/tokenizer/tokenizer.json
             ${OUTPUT_DIR}/tokenizer/tokenizer_config.json
  COMMAND ${PYTHON} buddy-pytorch-importer.py
            --repo-root ${REPO_ROOT}
            --output-dir ${OUTPUT_DIR}
            --compiler-build ${COMPILER_BUILD}
            --chip-pb ${CHIP_PB}
            --prefill-len ${PREFILL_LENGTH}
            --max-cache-len ${CACHE_LENGTH}
            --parts ${PARTS}
            --model meta-llama/Llama-3.2-3B-Instruct
  DEPENDS ${DESIGN_FILES}
          ${MODEL_DIR}/llama3-2-3b-instruct/prepare.py
          ${MODEL_DIR}/llama3-2-3b-instruct/configs/importer-param.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)
add_library(llama-application OBJECT EXCLUDE_FROM_ALL ${MODEL_DIR}/llama3-2-3b-instruct/main.cpp
  ${MODEL_DIR}/llama3-2-3b-instruct/generation.cpp
)
target_include_directories(llama-application PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(llama-application PUBLIC cxx_std_20)

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
add_library(llama3-2-3b-instruct-buckyball-poly-model STATIC EXCLUDE_FROM_ALL ${OBJECTS})
set_target_properties(llama3-2-3b-instruct-buckyball-poly-model PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
set_target_properties(llama3-2-3b-instruct-buckyball-poly-model PROPERTIES LINKER_LANGUAGE CXX)

add_executable(llama-run EXCLUDE_FROM_ALL execution.cpp worker.cpp loader.cpp step.cpp)
set_target_properties(llama-run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
add_dependencies(llama-run model-loader)
target_include_directories(llama-run PRIVATE ${MODEL_DIR}/llama3-2-3b-instruct
                                             ${CMAKE_CURRENT_SOURCE_DIR}
                                             ${ISA_DIR}/chip
                                             ${OUTPUT_DIR}
                                             ${BUDDY_SOURCE}/frontend/Interfaces
)
target_link_libraries(llama-run PRIVATE -static
                                        llama3-2-3b-instruct-buckyball-poly-model
                                        llama-application
                                        runtime_ant_tasks
                                        rvv_kernels_ant
                                        thread_runtime_ant
)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/llama.rax
  BYPRODUCTS ${OUTPUT_DIR}/layout.json
             ${OUTPUT_DIR}/llama.rhal.mlir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
  COMMAND ${PYTHON} pack.py
            --repo-root ${REPO_ROOT}
            --output-dir ${OUTPUT_DIR}
            --run-config ${CMAKE_CURRENT_SOURCE_DIR}/configs/running-param.toml
            --backend buckyball
            --rax-pack ${COMPILER_BUILD}/bin/rax-pack
  DEPENDS llama-run
          ${CMAKE_CURRENT_SOURCE_DIR}/loader/pack.py
          ${CMAKE_CURRENT_SOURCE_DIR}/loader/inputs.py
          ${CMAKE_CURRENT_SOURCE_DIR}/configs/running-param.toml
          ${OUTPUT_DIR}/model.json
          ${REPO_ROOT}/examples/models/poly/llama3-2-3b-instruct/loader/pack.py
          ${COMPILER_BUILD}/bin/rax-pack
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging llama3-2-3b-instruct model"
  VERBATIM
)
add_custom_target(llama3-2-3b-instruct-model-build DEPENDS ${OUTPUT_DIR}/llama.rax)
