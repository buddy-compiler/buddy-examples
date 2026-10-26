file(GLOB_RECURSE RECIPE_FRONTEND_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/frontend/*.py")

set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

if(NOT "matmul" IN_LIST DESIGN_TARGETS)
  message(FATAL_ERROR "Unknown BERT compiler target: matmul")
endif()
math(EXPR LAST_PARTITION "4 - 1")
set(BERT_SUBGRAPH_MLIR)
foreach(BERT_PARTITION RANGE 0 ${LAST_PARTITION})
  list(APPEND BERT_SUBGRAPH_MLIR ${OUTPUT_DIR}/subgraph${BERT_PARTITION}.mlir)
endforeach()

set(DESIGN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/trace/trace.toml")
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/forward.mlir
         ${BERT_SUBGRAPH_MLIR}
         ${OUTPUT_DIR}/arg0.data
         ${OUTPUT_DIR}/arg1.data
         ${OUTPUT_DIR}/vocab.txt
  COMMAND ${PYTHON} buddy-pytorch-importer.py
            --repo-root ${REPO_ROOT}
            --output-dir ${OUTPUT_DIR}
            --compiler-build ${COMPILER_BUILD}
            --design-pb ${DESIGN_PB}
            --partition-count 4
            --sequence-length 32
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          ${RECIPE_FRONTEND_SOURCES}
          ${DESIGN_FILES}
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          quant/calibration.py
          quant/quant.toml
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating BERT subgraphs and parameter files"
)

add_custom_command(
  OUTPUT forward.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUTPUT_DIR}/forward.mlir -buffer-deallocation-simplification -convert-linalg-to-loops -eliminate-empty-tensors -llvm-request-c-wrappers -convert-math-to-llvm -convert-math-to-libm -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -expand-strided-metadata -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts -canonicalize -cse -o ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUTPUT_DIR}/forward.mlir -buffer-deallocation-simplification -convert-linalg-to-loops -eliminate-empty-tensors -llvm-request-c-wrappers -convert-math-to-llvm -convert-math-to-libm -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -expand-strided-metadata -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts -canonicalize -cse |
          ${DESIGN_COMPILER}/bin/buddy-translate
            --buddy-to-llvmir |
          ${BUDDY_SOURCE}/llvm/build/bin/opt "-passes=function(sroa,early-cse,instcombine,simplifycfg,adce)" |
          ${DESIGN_COMPILER}/bin/buddy-llc -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o forward.o
  COMMAND ${CMAKE_OBJCOPY}
            --redefine-syms=${DESIGN_THREAD_BUILD_matmul}/kernels.rename
            --redefine-sym malloc=workspace_alloc
            --redefine-sym free=workspace_free forward.o
  DEPENDS "${OUTPUT_DIR}/forward.mlir"
          ${DESIGN_TARGET_PLUGIN_matmul}
          ${DESIGN_COMPILER}/bin/buddy-translate
          "${BUDDY_SOURCE}/llvm/build/bin/opt"
          ${DESIGN_COMPILER}/bin/buddy-llc
          ${DESIGN_THREAD_BUILD_matmul}/kernels.rename
  COMMENT "Building forward.o"
  VERBATIM
)
set(OBJECTS forward.o)
foreach(PARTITION RANGE 0 ${LAST_PARTITION})
  add_custom_command(
    OUTPUT subgraph${PARTITION}.o
    BYPRODUCTS ${OUTPUT_DIR}/subgraph${PARTITION}_linalg.mlir
               ${OUTPUT_DIR}/subgraph${PARTITION}_buckyball.mlir
               ${OUTPUT_DIR}/subgraph${PARTITION}_bank_ssa.mlir
               ${OUTPUT_DIR}/subgraph${PARTITION}_llvm.mlir
    COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUTPUT_DIR}/subgraph${PARTITION}.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" -o ${OUTPUT_DIR}/subgraph${PARTITION}_linalg.mlir
    COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUTPUT_DIR}/subgraph${PARTITION}.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
            ${DESIGN_TARGET_OPT_matmul} -eliminate-empty-tensors -convert-elementwise-to-linalg -convert-tensor-to-linalg -one-shot-bufferize=bufferize-function-boundaries -form-goban-mega-kernels -convert-linalg-to-buckyball -convert-linalg-to-toint8-ball -o ${OUTPUT_DIR}/subgraph${PARTITION}_buckyball.mlir
    COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUTPUT_DIR}/subgraph${PARTITION}.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
            ${DESIGN_TARGET_OPT_matmul} -eliminate-empty-tensors -convert-elementwise-to-linalg -convert-tensor-to-linalg -one-shot-bufferize=bufferize-function-boundaries -form-goban-mega-kernels -convert-linalg-to-buckyball -convert-linalg-to-toint8-ball -lower-matmul-region-to-bank-ssa -lower-buckyball-to-bank-ssa -assign-physical-banks -o ${OUTPUT_DIR}/subgraph${PARTITION}_bank_ssa.mlir
    COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUTPUT_DIR}/subgraph${PARTITION}.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
            ${DESIGN_TARGET_OPT_matmul} -eliminate-empty-tensors -convert-elementwise-to-linalg -convert-tensor-to-linalg -one-shot-bufferize=bufferize-function-boundaries -form-goban-mega-kernels -convert-linalg-to-buckyball -convert-linalg-to-toint8-ball -lower-matmul-region-to-bank-ssa -lower-buckyball-to-bank-ssa -assign-physical-banks -llvm-request-c-wrappers -lower-buckyball -convert-trace-to-llvm -expand-strided-metadata -convert-linalg-to-loops -lower-buckyball -lower-affine -canonicalize -cse -convert-scf-to-cf -convert-cf-to-llvm -buffer-deallocation-simplification -bufferization-lower-deallocations -convert-math-to-llvm -convert-math-to-libm -convert-arith-to-llvm -convert-func-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts -lower-to-stream -canonicalize -cse -o ${OUTPUT_DIR}/subgraph${PARTITION}_llvm.mlir
    COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUTPUT_DIR}/subgraph${PARTITION}.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
            ${DESIGN_TARGET_OPT_matmul} -eliminate-empty-tensors -convert-elementwise-to-linalg -convert-tensor-to-linalg -one-shot-bufferize=bufferize-function-boundaries -form-goban-mega-kernels -convert-linalg-to-buckyball -convert-linalg-to-toint8-ball -lower-matmul-region-to-bank-ssa -lower-buckyball-to-bank-ssa -assign-physical-banks -llvm-request-c-wrappers -lower-buckyball -convert-trace-to-llvm -expand-strided-metadata -convert-linalg-to-loops -lower-buckyball -lower-affine -canonicalize -cse -convert-scf-to-cf -convert-cf-to-llvm -buffer-deallocation-simplification -bufferization-lower-deallocations -convert-math-to-llvm -convert-math-to-libm -convert-arith-to-llvm -convert-func-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts -lower-to-stream -canonicalize -cse |
            ${DESIGN_COMPILER}/bin/buddy-translate
              --buddy-to-llvmir |
            ${BUDDY_SOURCE}/llvm/build/bin/opt "-passes=function(sroa,early-cse,instcombine,simplifycfg,adce)" |
            ${DESIGN_COMPILER}/bin/buddy-llc -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o subgraph${PARTITION}.o
    COMMAND ${CMAKE_OBJCOPY}
              --redefine-syms=${DESIGN_THREAD_BUILD_matmul}/kernels.rename
              --redefine-sym malloc=workspace_alloc
              --redefine-sym free=workspace_free subgraph${PARTITION}.o
    DEPENDS "${OUTPUT_DIR}/subgraph${PARTITION}.mlir"
            ${DESIGN_TARGET_PLUGIN_matmul}
            ${DESIGN_COMPILER}/bin/buddy-translate
            "${BUDDY_SOURCE}/llvm/build/bin/opt"
            ${DESIGN_COMPILER}/bin/buddy-llc
            ${DESIGN_THREAD_BUILD_matmul}/kernels.rename
    COMMENT "Building subgraph${PARTITION}.o"
    VERBATIM
  )
  list(APPEND OBJECTS subgraph${PARTITION}.o)
endforeach()
add_library(bert-base-uncased-emotion-buckyball-goban-model STATIC EXCLUDE_FROM_ALL ${OBJECTS})
set_target_properties(bert-base-uncased-emotion-buckyball-goban-model
  PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             LINKER_LANGUAGE CXX
)

add_library(bert-base-uncased-emotion-buckyball-goban-runner SHARED EXCLUDE_FROM_ALL
  ${CMAKE_CURRENT_SOURCE_DIR}/runner.cpp
)
set_target_properties(bert-base-uncased-emotion-buckyball-goban-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
)
target_include_directories(bert-base-uncased-emotion-buckyball-goban-runner
  PRIVATE ${OUTPUT_DIR}
          ${BUDDY_SOURCE}/frontend/Interfaces
)
target_link_libraries(bert-base-uncased-emotion-buckyball-goban-runner
  PRIVATE bert-base-uncased-emotion-buckyball-goban-model
          runtime-linux
          ${DESIGN_THREAD_BUILD_matmul}/libkernels.a
)

# The runner owns inference and the backend's native scheduling lifecycle.
get_filename_component(FLATBUFFERS_HEADER "${TOOLCHAIN}/include/flatbuffers/flatbuffers.h" REALPATH)
get_filename_component(FLATBUFFERS_INCLUDE "${FLATBUFFERS_HEADER}/../.." ABSOLUTE)
target_include_directories(bert-base-uncased-emotion-buckyball-goban-runner
  PRIVATE ${LIB_DIR}
          ${FLATBUFFERS_INCLUDE}
          ${BUDDY_SOURCE}/runtime/include
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/llvm/mlir/include
          ${OUTPUT_DIR}
          ${ISA_DIR}/design
          ${ISA_DIR}/matmul
          ${RUNTIME_DIR}/scheduler
          ${RUNTIME_DIR}/services
          ${RUNTIME_DIR}/../engine/include
)
set_target_properties(bert-base-uncased-emotion-buckyball-goban-runner
  PROPERTIES POSITION_INDEPENDENT_CODE ON
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
configure_file(bert-base-uncased-emotion.rhal.mlir.in ${OUTPUT_DIR}/bert-base-uncased-emotion.rhal.mlir @ONLY)

# Package the runner, model parameters and input metadata in one RAX.
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/bert-base-uncased-emotion.rax
  COMMAND ${COMPILER_BUILD}/bin/rax-pack ${OUTPUT_DIR}/bert-base-uncased-emotion.rhal.mlir
          --embed-payload
          -o ${OUTPUT_DIR}/bert-base-uncased-emotion.rax
  DEPENDS ${OUTPUT_DIR}/bert-base-uncased-emotion.rhal.mlir
          ${COMPILER_BUILD}/bin/rax-pack
          bert-base-uncased-emotion-buckyball-goban-runner
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
          ${OUTPUT_DIR}/vocab.txt
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging bert-base-uncased-emotion for buckyball-goban"
  VERBATIM
)
add_custom_target(bert-base-uncased-emotion-model-build DEPENDS ${OUTPUT_DIR}/bert-base-uncased-emotion.rax)
