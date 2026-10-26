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
          quant/calibration.py
          quant/quant.toml
          trace/trace.toml
          ${MODEL_DIR}/bert-base-uncased-emotion/output/bert-base-uncased-emotion/model.safetensors
          ${MODEL_DIR}/bert-base-uncased-emotion/model.py
          ${MODEL_DIR}/bert-base-uncased-emotion/output/bert-base-uncased-emotion/config.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing bert-base-uncased-emotion for nv"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Compile the host entry point. Each command must succeed before the next runs.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_trace.mlir
             ${OUTPUT_DIR}/forward_linalg.mlir
             ${OUTPUT_DIR}/forward_gpu.mlir
             ${OUTPUT_DIR}/forward_gpu_llvm.mlir
             ${OUTPUT_DIR}/forward_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/forward_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/forward_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_linalg.mlir
            -pass-pipeline "builtin.module(func.func(buffer-hoisting),canonicalize,cse,expand-strided-metadata,func.func(convert-memcpy-to-gpu,gpu-async-region))"
            -o ${OUTPUT_DIR}/forward_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_gpu.mlir
            -llvm-request-c-wrappers
            -gpu-to-llvm
            -o ${OUTPUT_DIR}/forward_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/forward_gpu_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/forward_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward_gpu.ll -filetype=obj -relocation-model=pic -O2 -o forward_gpu.o
  DEPENDS ${OUTPUT_DIR}/forward.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling bert-base-uncased-emotion host entry point"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower model computation to GPU kernels and embed the NVIDIA binary.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_trace.mlir
             ${OUTPUT_DIR}/subgraph0_gpu_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_bufferized.mlir
             ${OUTPUT_DIR}/subgraph0_copies.mlir
             ${OUTPUT_DIR}/subgraph0_gpu_gpu.mlir
             ${OUTPUT_DIR}/subgraph0_gpu_async.mlir
             ${OUTPUT_DIR}/subgraph0_gpu_llvm.mlir
             ${OUTPUT_DIR}/subgraph0_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/subgraph0_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_gpu_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_gpu_linalg.mlir
            -convert-elementwise-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -o ${OUTPUT_DIR}/subgraph0_bufferized.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_bufferized.mlir
            -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))"
            -o ${OUTPUT_DIR}/subgraph0_copies.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_copies.mlir
            -convert-linalg-to-parallel-loops
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse
            -o ${OUTPUT_DIR}/subgraph0_gpu_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_gpu_gpu.mlir
            -pass-pipeline "builtin.module(func.func(convert-memcpy-to-gpu{process-args=0},gpu-async-region),canonicalize)"
            -o ${OUTPUT_DIR}/subgraph0_gpu_async.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_gpu_async.mlir
            --nvvm-reflect-add=__CUDA_PREC_SQRT=1
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=${NV_CUBIN_VERSION} cubin-features=+ptx71 cubin-format=fatbin ptxas-cmd-options=--fmad=false"
            -o ${OUTPUT_DIR}/subgraph0_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_gpu_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/subgraph0_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0_gpu.ll -filetype=obj -relocation-model=pic -O2 -o subgraph0_gpu.o
  DEPENDS ${OUTPUT_DIR}/subgraph0.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling bert-base-uncased-emotion NVIDIA kernels"
  VERBATIM
)

add_library(bert-base-uncased-emotion-nv-model SHARED EXCLUDE_FROM_ALL forward_gpu.o subgraph0_gpu.o)
set_target_properties(bert-base-uncased-emotion-nv-model
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             LINKER_LANGUAGE CXX
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_link_directories(bert-base-uncased-emotion-nv-model
  PRIVATE ${BUDDY_SOURCE}/llvm/build/lib
)
target_link_libraries(bert-base-uncased-emotion-nv-model
  PRIVATE nv-runtime
          mlir_c_runner_utils
          mlir_runner_utils
          mlir_async_runtime
          mlir_cuda_runtime
          dl
          pthread
)

# ===----------------------------------------------------------------------===
# Build the Buddy InferenceRunner plugin.
# ===----------------------------------------------------------------------===
add_library(bert-base-uncased-emotion-nv-runner SHARED EXCLUDE_FROM_ALL runner.cpp)
set_target_properties(bert-base-uncased-emotion-nv-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_include_directories(bert-base-uncased-emotion-nv-runner
  PRIVATE ${LIB_DIR}
          ${OUTPUT_DIR}
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/runtime/include
          ${COMPILER_BUILD}/runtime/include
)
target_link_directories(bert-base-uncased-emotion-nv-runner
  PRIVATE ${BUDDY_SOURCE}/llvm/build/lib
)
target_link_libraries(bert-base-uncased-emotion-nv-runner
  PRIVATE dl
)

# ===----------------------------------------------------------------------===
# Package the RAX
#
# Contents:
# model library (libbert-base-uncased-emotion-nv-model.so),
# runner library (libbert-base-uncased-emotion-nv-runner.so),
# FP32 parameters (arg0.data),
# INT8 weights (arg1.data),
# token vocabulary (vocab.txt),
# MLIR runtime libraries (libmlir_c_runner_utils.so, libmlir_runner_utils.so,
# libmlir_async_runtime.so, libmlir_cuda_runtime.so).
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
          bert-base-uncased-emotion-nv-model
          bert-base-uncased-emotion-nv-runner
          ${COMPILER_BUILD}/bin/rax-pack
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
          ${MODEL_DIR}/bert-base-uncased-emotion/vocab.txt
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_c_runner_utils.so
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_runner_utils.so
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_async_runtime.so
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_cuda_runtime.so
  COMMENT "Packaging NVIDIA bert-base-uncased-emotion as RAX"
  VERBATIM
)
add_custom_target(buddy-nv-bert-base-uncased-emotion-rax-build DEPENDS ${OUTPUT_DIR}/bert-base-uncased-emotion.rax)

add_subdirectory(origin)
