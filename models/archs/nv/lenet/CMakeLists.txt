file(GLOB_RECURSE RECIPE_FRONTEND_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/frontend/*.py")

set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

# ===----------------------------------------------------------------------===
# Import the trained model and export parameters.
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
          frontend/permute/graph.py
          trace/trace.toml
          ${MODEL_DIR}/lenet/model.py
          ${MODEL_DIR}/lenet/output/lenet/model.pth
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing LeNet for NVIDIA"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Compile the host entry point. Each command must succeed before the next runs.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/forward_gpu.o
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
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward_gpu.ll -filetype=obj -relocation-model=pic -O2 -o ${OUTPUT_DIR}/forward_gpu.o
  DEPENDS ${OUTPUT_DIR}/forward.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling LeNet host entry point"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower model computation to GPU kernels and embed the NVIDIA binary.
# Bufferize trace operands first; lower GPU traces after device allocation
# conversion, then synchronize kernels and host copies with gpu-async-region.
# Keep single-iteration parallel loops until GPU mapping so scalar quantization
# and reductions never access device allocations from the host.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/subgraph0_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_trace.mlir
             ${OUTPUT_DIR}/subgraph0_gpu_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_bufferized.mlir
             ${OUTPUT_DIR}/subgraph0_copies.mlir
             ${OUTPUT_DIR}/subgraph0_gpu_gpu.mlir
             ${OUTPUT_DIR}/subgraph0_gpu_memory.mlir
             ${OUTPUT_DIR}/subgraph0_gpu_async.mlir
             ${OUTPUT_DIR}/subgraph0_gpu_llvm.mlir
             ${OUTPUT_DIR}/subgraph0_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
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
            -pass-pipeline "builtin.module(func.func(convert-memcpy-to-gpu{process-args=0}),canonicalize)"
            -o ${OUTPUT_DIR}/subgraph0_gpu_memory.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_gpu_memory.mlir
            -convert-trace-to-llvm=gpu
            -o ${OUTPUT_DIR}/subgraph0_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_trace.mlir
            -pass-pipeline "builtin.module(func.func(gpu-async-region),canonicalize)"
            -o ${OUTPUT_DIR}/subgraph0_gpu_async.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_gpu_async.mlir
            --nvvm-reflect-add=__CUDA_PREC_SQRT=1
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=${NV_CUBIN_VERSION} cubin-features=+ptx71 cubin-format=fatbin ptxas-cmd-options=--fmad=false"
            -o ${OUTPUT_DIR}/subgraph0_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_gpu_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/subgraph0_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0_gpu.ll -filetype=obj -relocation-model=pic -O2 -o ${OUTPUT_DIR}/subgraph0_gpu.o
  DEPENDS ${OUTPUT_DIR}/subgraph0.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling LeNet NVIDIA kernels"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Host library containing the compiled NVIDIA kernels.
# ===----------------------------------------------------------------------===
add_library(lenet-nv-model SHARED EXCLUDE_FROM_ALL ${OUTPUT_DIR}/subgraph0_gpu.o ${OUTPUT_DIR}/forward_gpu.o)
set_target_properties(lenet-nv-model
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             LINKER_LANGUAGE CXX
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_link_directories(lenet-nv-model
  PRIVATE ${BUDDY_SOURCE}/llvm/build/lib
)
target_link_libraries(lenet-nv-model
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
add_library(lenet-nv-runner SHARED EXCLUDE_FROM_ALL runner.cpp)
set_target_properties(lenet-nv-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_include_directories(lenet-nv-runner
  PRIVATE ${LIB_DIR}
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/runtime/include
          ${COMPILER_BUILD}/runtime/include
)
target_link_libraries(lenet-nv-runner
  PRIVATE dl
          PNG::PNG
)

# ===----------------------------------------------------------------------===
# Package the RAX
#
# Contents:
# model library (liblenet-nv-model.so),
# runner library (liblenet-nv-runner.so),
# FP32 parameters (arg0.data),
# INT8 weights (arg1.data),
# MLIR runtime libraries (libmlir_c_runner_utils.so, libmlir_runner_utils.so,
# libmlir_async_runtime.so, libmlir_cuda_runtime.so).
#
# Output: output/lenet.rax embeds these files alongside RHAL metadata that
# identifies the runner, model library, parameters and runtime dependencies.
#
# ===----------------------------------------------------------------------===
configure_file(lenet.rhal.mlir.in ${OUTPUT_DIR}/lenet.rhal.mlir @ONLY)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/lenet.rax
  COMMAND ${COMPILER_BUILD}/bin/rax-pack ${OUTPUT_DIR}/lenet.rhal.mlir
          -o ${OUTPUT_DIR}/lenet.rax
          --embed-payload
  DEPENDS ${OUTPUT_DIR}/lenet.rhal.mlir
          lenet-nv-model
          lenet-nv-runner
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
          ${COMPILER_BUILD}/bin/rax-pack
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_c_runner_utils.so
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_runner_utils.so
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_async_runtime.so
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_cuda_runtime.so
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging NVIDIA LeNet as RAX"
  VERBATIM
)
add_custom_target(buddy-nv-lenet-rax-build DEPENDS ${OUTPUT_DIR}/lenet.rax)

add_subdirectory(origin)
