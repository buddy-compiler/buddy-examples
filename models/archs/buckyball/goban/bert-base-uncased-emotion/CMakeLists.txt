set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

if(NOT "goban" IN_LIST TARGETS)
  message(FATAL_ERROR "Unknown BERT compiler target: goban")
endif()
math(EXPR LAST_PARTITION "4 - 1")
set(BERT_SUBGRAPH_MLIR)
foreach(BERT_PARTITION RANGE 0 ${LAST_PARTITION})
  list(
    APPEND BERT_SUBGRAPH_MLIR
    ${OUTPUT_DIR}/subgraph${BERT_PARTITION}.mlir
    ${OUTPUT_DIR}/subgraph${BERT_PARTITION}-task.c
  )
endforeach()

file(GLOB_RECURSE DESIGN_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/quant/*"
  "${CMAKE_CURRENT_SOURCE_DIR}/trace/*" "${CMAKE_CURRENT_SOURCE_DIR}/permute/*"
)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/forward.mlir
         ${BERT_SUBGRAPH_MLIR}
         ${OUTPUT_DIR}/params.f32
         ${OUTPUT_DIR}/weights.bin
         ${OUTPUT_DIR}/scales.bin
         ${OUTPUT_DIR}/bert-parameters.h
  COMMAND ${PYTHON} buddy-pytorch-importer.py
            --repo-root ${REPO_ROOT}
            --output-dir ${OUTPUT_DIR}
            --compiler-build ${COMPILER_BUILD}
            --chip-pb ${CHIP_PB}
            --partition-count 4
            --sequence-length 32
  DEPENDS ${DESIGN_FILES}
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${MODEL_DIR}/bert-base-uncased-emotion/configs/importer-param.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating BERT subgraphs and parameter files"
)

add_custom_command(
  OUTPUT forward.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${COMPILER_BUILD_goban}/bin/buddy-opt ${OUTPUT_DIR}/forward.mlir
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
            -o ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${COMPILER_BUILD_goban}/bin/buddy-opt ${OUTPUT_DIR}/forward.mlir
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
          ${COMPILER_BUILD_goban}/bin/buddy-translate
            --buddy-to-llvmir |
          ${BUDDY_SOURCE}/llvm/build/bin/opt "-passes=function(sroa,early-cse,instcombine,simplifycfg,adce)" |
          ${COMPILER_BUILD_goban}/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o forward.o
  COMMAND ${CMAKE_OBJCOPY}
            --redefine-syms=${THREAD_RUNTIME_DIR}/goban.rename
            --redefine-sym malloc=workspace_alloc
            --redefine-sym free=workspace_free forward.o
  DEPENDS "${OUTPUT_DIR}/forward.mlir"
          ${COMPILER_BUILD_goban}/bin/buddy-opt
          ${COMPILER_BUILD_goban}/bin/buddy-translate
          "${BUDDY_SOURCE}/llvm/build/bin/opt"
          ${COMPILER_BUILD_goban}/bin/buddy-llc
          ${THREAD_RUNTIME_DIR}/goban.rename
  COMMENT "Building forward.o"
  VERBATIM
)
set(OBJECTS forward.o)
foreach(PARTITION RANGE 0 ${LAST_PARTITION})
  add_custom_command(
    OUTPUT subgraph${PARTITION}.o
    BYPRODUCTS ${OUTPUT_DIR}/subgraph${PARTITION}_linalg.mlir
               ${OUTPUT_DIR}/subgraph${PARTITION}_tile.mlir
               ${OUTPUT_DIR}/subgraph${PARTITION}_buckyball.mlir
               ${OUTPUT_DIR}/subgraph${PARTITION}_bank_ssa.mlir
               ${OUTPUT_DIR}/subgraph${PARTITION}_llvm.mlir
    COMMAND ${COMPILER_BUILD_goban}/bin/buddy-opt ${OUTPUT_DIR}/subgraph${PARTITION}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
              -o ${OUTPUT_DIR}/subgraph${PARTITION}_linalg.mlir
    COMMAND ${COMPILER_BUILD_goban}/bin/buddy-opt ${OUTPUT_DIR}/subgraph${PARTITION}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
            ${COMPILER_BUILD_goban}/bin/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -one-shot-bufferize=bufferize-function-boundaries
              -convert-linalg-to-tile=quantized-only
              -o ${OUTPUT_DIR}/subgraph${PARTITION}_tile.mlir
    COMMAND ${COMPILER_BUILD_goban}/bin/buddy-opt ${OUTPUT_DIR}/subgraph${PARTITION}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
            ${COMPILER_BUILD_goban}/bin/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -one-shot-bufferize=bufferize-function-boundaries
              -convert-linalg-to-tile=quantized-only
              -convert-tile-to-buckyball
              -o ${OUTPUT_DIR}/subgraph${PARTITION}_buckyball.mlir
    COMMAND ${COMPILER_BUILD_goban}/bin/buddy-opt ${OUTPUT_DIR}/subgraph${PARTITION}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
            ${COMPILER_BUILD_goban}/bin/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -one-shot-bufferize=bufferize-function-boundaries
              -convert-linalg-to-tile=quantized-only
              -convert-tile-to-buckyball
              -lower-buckyball-to-bank-ssa
              -assign-physical-banks
              -o ${OUTPUT_DIR}/subgraph${PARTITION}_bank_ssa.mlir
    COMMAND ${COMPILER_BUILD_goban}/bin/buddy-opt ${OUTPUT_DIR}/subgraph${PARTITION}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
            ${COMPILER_BUILD_goban}/bin/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -one-shot-bufferize=bufferize-function-boundaries
              -convert-linalg-to-tile=quantized-only
              -convert-tile-to-buckyball
              -lower-buckyball-to-bank-ssa
              -assign-physical-banks
              -llvm-request-c-wrappers
              -lower-bank-ssa-to-intrinsics
              -convert-trace-to-llvm
              -expand-strided-metadata
              -convert-linalg-to-loops
              -lower-buckyball
              -lower-affine
              -canonicalize
              -cse
              -convert-scf-to-cf
              -convert-cf-to-llvm
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
              -o ${OUTPUT_DIR}/subgraph${PARTITION}_llvm.mlir
    COMMAND ${COMPILER_BUILD_goban}/bin/buddy-opt ${OUTPUT_DIR}/subgraph${PARTITION}.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
            ${COMPILER_BUILD_goban}/bin/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -one-shot-bufferize=bufferize-function-boundaries
              -convert-linalg-to-tile=quantized-only
              -convert-tile-to-buckyball
              -lower-buckyball-to-bank-ssa
              -assign-physical-banks
              -llvm-request-c-wrappers
              -lower-bank-ssa-to-intrinsics
              -convert-trace-to-llvm
              -expand-strided-metadata
              -convert-linalg-to-loops
              -lower-buckyball
              -lower-affine
              -canonicalize
              -cse
              -convert-scf-to-cf
              -convert-cf-to-llvm
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
            ${COMPILER_BUILD_goban}/bin/buddy-translate
              --buddy-to-llvmir |
            ${BUDDY_SOURCE}/llvm/build/bin/opt "-passes=function(sroa,early-cse,instcombine,simplifycfg,adce)" |
            ${COMPILER_BUILD_goban}/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o subgraph${PARTITION}.o
    COMMAND ${CMAKE_OBJCOPY}
              --redefine-syms=${THREAD_RUNTIME_DIR}/goban.rename
              --redefine-sym malloc=workspace_alloc
              --redefine-sym free=workspace_free subgraph${PARTITION}.o
    DEPENDS "${OUTPUT_DIR}/subgraph${PARTITION}.mlir"
            ${COMPILER_BUILD_goban}/bin/buddy-opt
            ${COMPILER_BUILD_goban}/bin/buddy-translate
            "${BUDDY_SOURCE}/llvm/build/bin/opt"
            ${COMPILER_BUILD_goban}/bin/buddy-llc
            ${THREAD_RUNTIME_DIR}/goban.rename
    COMMENT "Building subgraph${PARTITION}.o"
    VERBATIM
  )
  list(APPEND OBJECTS subgraph${PARTITION}.o ${OUTPUT_DIR}/subgraph${PARTITION}-task.c)
endforeach()
add_library(bert-base-uncased-emotion-buckyball-goban-model STATIC EXCLUDE_FROM_ALL ${OBJECTS})
set_target_properties(bert-base-uncased-emotion-buckyball-goban-model PROPERTIES ARCHIVE_OUTPUT_DIRECTORY
  "${OUTPUT_DIR}"
)
set_target_properties(bert-base-uncased-emotion-buckyball-goban-model PROPERTIES LINKER_LANGUAGE CXX)

add_executable(bert-run EXCLUDE_FROM_ALL ${MODEL_DIR}/bert-base-uncased-emotion/main.cpp)
set_target_properties(bert-run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
add_dependencies(bert-run model-loader)
target_include_directories(bert-run PRIVATE ${OUTPUT_DIR}
                                            ${BUDDY_SOURCE}/frontend/Interfaces
)
target_link_libraries(bert-run PRIVATE -static
                                       bert-base-uncased-emotion-buckyball-goban-model
                                       runtime_linux
                                       rvv_kernels
)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/bert.rax
  BYPRODUCTS ${OUTPUT_DIR}/layout.json
             ${OUTPUT_DIR}/bert.rhal.mlir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
  COMMAND ${PYTHON} pack.py
            --repo-root ${REPO_ROOT}
            --model-dir ${OUTPUT_DIR}
            --entrypoint examples.models.goban.bert-base-uncased-emotion.serve:run
            --backend buckyball
            --task-runtime linux
            --rax-pack ${COMPILER_BUILD}/bin/rax-pack
  DEPENDS bert-run
          ${CMAKE_CURRENT_SOURCE_DIR}/pack.py
          ${OUTPUT_DIR}/bert.rax
          ${OUTPUT_DIR}/forward.mlir
          ${BERT_SUBGRAPH_MLIR}
          ${OUTPUT_DIR}/params.f32
          ${OUTPUT_DIR}/weights.bin
          ${OUTPUT_DIR}/scales.bin
          ${OUTPUT_DIR}/bert-parameters.h
          ${COMPILER_BUILD}/bin/rax-pack
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging bert-base-uncased-emotion model"
  VERBATIM
)
add_custom_target(bert-base-uncased-emotion-model-build DEPENDS ${OUTPUT_DIR}/bert.rax)
