file(GLOB_RECURSE RECIPE_FRONTEND_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/frontend/*.py")

set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

add_custom_command(
  OUTPUT ${OUTPUT_DIR}/forward.mlir
         ${OUTPUT_DIR}/subgraph0.mlir
         ${OUTPUT_DIR}/arg0.data
         ${OUTPUT_DIR}/arg1.data
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
  COMMAND ${PYTHON} buddy-pytorch-importer.py
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          ${RECIPE_FRONTEND_SOURCES}
          buddy-pytorch-importer.py
          quant/quant.toml
          trace/trace.toml
          ${MODEL_DIR}/bert-base-uncased-emotion/output/bert-base-uncased-emotion/model.safetensors
          ${MODEL_DIR}/bert-base-uncased-emotion/model.py
          ${MODEL_DIR}/bert-base-uncased-emotion/output/bert-base-uncased-emotion/config.json
          ${CMAKE_CURRENT_SOURCE_DIR}/trace/trace.toml
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing bert-base-uncased-emotion for x86"
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
  COMMENT "Compiling bert-base-uncased-emotion forward for x86"
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
  COMMENT "Compiling bert-base-uncased-emotion subgraph0 for x86"
  VERBATIM
)

add_library(bert-base-uncased-emotion-x86-model SHARED EXCLUDE_FROM_ALL forward.o subgraph0.o)
set_target_properties(bert-base-uncased-emotion-x86-model
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             LINKER_LANGUAGE CXX
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_link_directories(bert-base-uncased-emotion-x86-model
  PRIVATE ${BUDDY_SOURCE}/llvm/build/lib
)
target_link_libraries(bert-base-uncased-emotion-x86-model
  PRIVATE x86-runtime
          mlir_c_runner_utils
          mlir_runner_utils
          dl
          pthread
)

# ===----------------------------------------------------------------------===
# Build the Buddy InferenceRunner plugin.
# ===----------------------------------------------------------------------===
add_library(bert-base-uncased-emotion-x86-runner SHARED EXCLUDE_FROM_ALL runner.cpp)
set_target_properties(bert-base-uncased-emotion-x86-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_include_directories(bert-base-uncased-emotion-x86-runner
  PRIVATE ${LIB_DIR}
          ${OUTPUT_DIR}
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/runtime/include
          ${COMPILER_BUILD}/runtime/include
)
target_link_directories(bert-base-uncased-emotion-x86-runner
  PRIVATE ${BUDDY_SOURCE}/llvm/build/lib
)
target_link_libraries(bert-base-uncased-emotion-x86-runner
  PRIVATE dl
)

# ===----------------------------------------------------------------------===
# Package the RAX
#
# Contents:
# model library (libbert-base-uncased-emotion-x86-model.so),
# runner library (libbert-base-uncased-emotion-x86-runner.so),
# FP32 parameters (arg0.data),
# INT8 weights (arg1.data),
# token vocabulary (vocab.txt),
# MLIR runtime libraries (libmlir_c_runner_utils.so, libmlir_runner_utils.so).
#
# Output: output/bert-base-uncased-emotion.rax embeds these files alongside RHAL metadata that
# identifies the runner, model library, parameters and runtime dependencies.
#
# ===----------------------------------------------------------------------===
configure_file(bert-base-uncased-emotion.rhal.mlir.in ${OUTPUT_DIR}/bert-base-uncased-emotion.rhal.mlir @ONLY)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/bert-base-uncased-emotion.rax
  COMMAND ${COMPILER_BUILD}/bin/rax-pack ${OUTPUT_DIR}/bert-base-uncased-emotion.rhal.mlir
          -o ${OUTPUT_DIR}/bert-base-uncased-emotion.rax
          --embed-payload
  DEPENDS ${OUTPUT_DIR}/bert-base-uncased-emotion.rhal.mlir
          bert-base-uncased-emotion-x86-model
          bert-base-uncased-emotion-x86-runner
          ${COMPILER_BUILD}/bin/rax-pack
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
          ${MODEL_DIR}/bert-base-uncased-emotion/vocab.txt
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_c_runner_utils.so
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_runner_utils.so
  COMMENT "Packaging x86 bert-base-uncased-emotion as RAX"
  VERBATIM
)
add_custom_target(buddy-x86-bert-base-uncased-emotion-rax-build DEPENDS ${OUTPUT_DIR}/bert-base-uncased-emotion.rax)

add_subdirectory(origin)
