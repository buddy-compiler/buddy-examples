file(GLOB_RECURSE RECIPE_FRONTEND_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/frontend/*.py")

set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

# ===----------------------------------------------------------------------===
# Import the model with offline INT8 weights and online activation scales.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/forward.mlir
         ${OUTPUT_DIR}/subgraph0.mlir
         ${OUTPUT_DIR}/arg0.data
         ${OUTPUT_DIR}/arg1.data
  COMMAND ${PYTHON} buddy-pytorch-importer.py
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          ${RECIPE_FRONTEND_SOURCES}
          buddy-pytorch-importer.py
          quant/quant.toml
          trace/trace.toml
          ${MODEL_DIR}/bert-base-uncased-emotion/output/bert-base-uncased-emotion/model.safetensors
          ${MODEL_DIR}/bert-base-uncased-emotion/model.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower forward for the standalone Gemmini target.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_linalg.mlir
             ${OUTPUT_DIR}/forward_llvm.mlir
             ${OUTPUT_DIR}/forward.ll
  COMMAND ${GEMMINI_OPT} ${OUTPUT_DIR}/forward.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" -o ${OUTPUT_DIR}/forward_linalg.mlir
  COMMAND ${GEMMINI_OPT} ${OUTPUT_DIR}/forward_linalg.mlir -llvm-request-c-wrappers -convert-elementwise-to-linalg -convert-trace-to-llvm -expand-strided-metadata -convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/forward_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/forward.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+xgemmini,+D -float-abi=hard -o forward.o
  DEPENDS ${OUTPUT_DIR}/forward.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  VERBATIM
  DEPENDS ${GEMMINI_PLUGIN}
)

# ===----------------------------------------------------------------------===
# Lower subgraph0 for the standalone Gemmini target.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_llvm.mlir
             ${OUTPUT_DIR}/subgraph0.ll
  COMMAND ${GEMMINI_OPT} ${OUTPUT_DIR}/subgraph0.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" -o ${OUTPUT_DIR}/subgraph0_linalg.mlir
  COMMAND ${GEMMINI_OPT} ${OUTPUT_DIR}/subgraph0_linalg.mlir -llvm-request-c-wrappers -convert-elementwise-to-linalg -one-shot-bufferize=bufferize-function-boundaries -ownership-based-buffer-deallocation -buffer-deallocation-simplification -bufferization-lower-deallocations -prepare-gemmini-int8 -convert-linalg-to-gemmini -convert-trace-to-llvm -expand-strided-metadata -convert-linalg-to-loops -lower-gemmini -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/subgraph0.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+xgemmini,+D -float-abi=hard -o subgraph0.o
  DEPENDS ${OUTPUT_DIR}/subgraph0.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  VERBATIM
  DEPENDS ${GEMMINI_PLUGIN}
)

# ===----------------------------------------------------------------------===
# Link the model and runner into a static RISC-V executable.
# ===----------------------------------------------------------------------===
add_library(bert-base-uncased-emotion-gemmini-model SHARED EXCLUDE_FROM_ALL forward.o subgraph0.o)
set_target_properties(bert-base-uncased-emotion-gemmini-model
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             LINKER_LANGUAGE CXX
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
set_target_properties(gemmini-runtime
  PROPERTIES POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(bert-base-uncased-emotion-gemmini-model
  PRIVATE gemmini-runtime
)

# ===----------------------------------------------------------------------===
# Build the Buddy InferenceRunner plugin for RISC-V Linux.
# ===----------------------------------------------------------------------===
get_filename_component(FLATBUFFERS_HEADER "${TOOLCHAIN}/include/flatbuffers/flatbuffers.h" REALPATH)
get_filename_component(FLATBUFFERS_INCLUDE "${FLATBUFFERS_HEADER}/../.." ABSOLUTE)
add_library(bert-base-uncased-emotion-gemmini-runner SHARED EXCLUDE_FROM_ALL runner.cpp)
set_target_properties(bert-base-uncased-emotion-gemmini-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_include_directories(bert-base-uncased-emotion-gemmini-runner
  PRIVATE ${LIB_DIR}
          ${FLATBUFFERS_INCLUDE}
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/runtime/include
          ${COMPILER_BUILD}/runtime/include
)
target_link_libraries(bert-base-uncased-emotion-gemmini-runner
  PRIVATE dl
)

# ===----------------------------------------------------------------------===
# Package the RAX
#
# Contents:
# RISC-V model library (libbert-base-uncased-emotion-gemmini-model.so),
# RISC-V runner library (libbert-base-uncased-emotion-gemmini-runner.so),
# FP32 parameters (arg0.data), and
# INT8 weights (arg1.data).
#
# Output: output/bert-base-uncased-emotion.rax embeds these files alongside RHAL metadata
# identifying the runner, model entry and parameter buffers.
# ===----------------------------------------------------------------------===
configure_file(bert-base-uncased-emotion.rhal.mlir.in ${OUTPUT_DIR}/bert-base-uncased-emotion.rhal.mlir @ONLY)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/bert-base-uncased-emotion.rax
  COMMAND ${COMPILER_BUILD}/bin/rax-pack ${OUTPUT_DIR}/bert-base-uncased-emotion.rhal.mlir
          --embed-payload
          -o ${OUTPUT_DIR}/bert-base-uncased-emotion.rax
  DEPENDS ${OUTPUT_DIR}/bert-base-uncased-emotion.rhal.mlir
          ${COMPILER_BUILD}/bin/rax-pack
          bert-base-uncased-emotion-gemmini-model
          bert-base-uncased-emotion-gemmini-runner
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
          ${MODEL_DIR}/bert-base-uncased-emotion/vocab.txt
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging bert-base-uncased-emotion for gemmini"
  VERBATIM
)
add_custom_target(bert-base-uncased-emotion-model-build DEPENDS ${OUTPUT_DIR}/bert-base-uncased-emotion.rax)
