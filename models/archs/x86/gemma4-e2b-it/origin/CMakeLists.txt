file(GLOB MODEL_WEIGHTS CONFIGURE_DEPENDS "${MODEL_DIR}/gemma4-e2b-it/output/gemma4-e2b-it/*.safetensors"
  "${MODEL_DIR}/gemma4-e2b-it/output/gemma4-e2b-it/*.bin"
)

set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

add_custom_command(
  OUTPUT ${OUTPUT_DIR}/forward.mlir
         ${OUTPUT_DIR}/subgraph0.mlir
         ${OUTPUT_DIR}/arg0.data
         ${OUTPUT_DIR}/vocab.txt
  COMMAND ${PYTHON} buddy-pytorch-importer.py
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          buddy-pytorch-importer.py
          ${MODEL_WEIGHTS}
          trace/trace.toml
          ${MODEL_DIR}/gemma4-e2b-it/output/gemma4-e2b-it/config.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing gemma4-e2b-it for x86"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Compile the host entry point to native x86 code.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_trace.mlir
             ${OUTPUT_DIR}/forward_linalg.mlir
             ${OUTPUT_DIR}/forward_llvm.mlir
             ${OUTPUT_DIR}/forward.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/forward_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/forward_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_linalg.mlir
            -convert-elementwise-to-linalg
            -convert-linalg-to-loops
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/forward_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/forward.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward.ll -filetype=obj -relocation-model=pic -O3 -o forward.o
  DEPENDS ${OUTPUT_DIR}/forward.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling gemma4-e2b-it forward for x86"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower model computation to native x86 code.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_trace.mlir
             ${OUTPUT_DIR}/subgraph0_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_llvm.mlir
             ${OUTPUT_DIR}/subgraph0.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/subgraph0_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_linalg.mlir
            -convert-elementwise-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-linalg-to-loops
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/subgraph0.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0.ll -filetype=obj -relocation-model=pic -O3 -o subgraph0.o
  DEPENDS ${OUTPUT_DIR}/subgraph0.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling gemma4-e2b-it subgraph0 for x86"
  VERBATIM
)

add_library(gemma4-e2b-it-x86-origin-model SHARED EXCLUDE_FROM_ALL forward.o subgraph0.o)
set_target_properties(gemma4-e2b-it-x86-origin-model
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             LINKER_LANGUAGE CXX
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_link_directories(gemma4-e2b-it-x86-origin-model
  PRIVATE ${BUDDY_SOURCE}/llvm/build/lib
)
target_link_libraries(gemma4-e2b-it-x86-origin-model
  PRIVATE x86-runtime
          mlir_c_runner_utils
          mlir_runner_utils
          dl
          pthread
)

# ===----------------------------------------------------------------------===
# Build the Buddy InferenceRunner plugin.
# ===----------------------------------------------------------------------===
add_library(gemma4-e2b-it-x86-origin-runner SHARED EXCLUDE_FROM_ALL runner.cpp)
set_target_properties(gemma4-e2b-it-x86-origin-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_include_directories(gemma4-e2b-it-x86-origin-runner
  PRIVATE ${LIB_DIR}
          ${OUTPUT_DIR}
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/runtime/include
          ${COMPILER_BUILD}/runtime/include
)
target_link_directories(gemma4-e2b-it-x86-origin-runner
  PRIVATE ${BUDDY_SOURCE}/llvm/build/lib
)
target_link_libraries(gemma4-e2b-it-x86-origin-runner
  PRIVATE dl
)

# ===----------------------------------------------------------------------===
# Package the RAX
#
# Contents:
# model library (libgemma4-e2b-it-x86-origin-model.so),
# runner library (libgemma4-e2b-it-x86-origin-runner.so),
# FP32 parameters (arg0.data),
# token vocabulary (vocab.txt),
# MLIR runtime libraries (libmlir_c_runner_utils.so, libmlir_runner_utils.so).
#
# Output: output/gemma4-e2b-it.rax embeds these files alongside RHAL metadata that
# identifies the runner, model library, parameters and runtime dependencies.
#
# ===----------------------------------------------------------------------===
configure_file(gemma4-e2b-it.rhal.mlir.in ${OUTPUT_DIR}/gemma4-e2b-it.rhal.mlir @ONLY)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/gemma4-e2b-it.rax
  COMMAND ${COMPILER_BUILD}/bin/rax-pack ${OUTPUT_DIR}/gemma4-e2b-it.rhal.mlir
          -o ${OUTPUT_DIR}/gemma4-e2b-it.rax
          --embed-payload
  DEPENDS ${OUTPUT_DIR}/gemma4-e2b-it.rhal.mlir
          gemma4-e2b-it-x86-origin-model
          gemma4-e2b-it-x86-origin-runner
          ${COMPILER_BUILD}/bin/rax-pack
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/vocab.txt
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_c_runner_utils.so
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_runner_utils.so
  COMMENT "Packaging x86 gemma4-e2b-it as RAX"
  VERBATIM
)
add_custom_target(buddy-x86-gemma4-e2b-it-rax-origin-build DEPENDS ${OUTPUT_DIR}/gemma4-e2b-it.rax)
