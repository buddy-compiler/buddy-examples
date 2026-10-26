# ===----------------------------------------------------------------------===
# Original FP32 baseline, with independent objects, libraries and RAX output.
# ===----------------------------------------------------------------------===
set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

# ===----------------------------------------------------------------------===
# Import the trained model and export parameters.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/forward.mlir
         ${OUTPUT_DIR}/subgraph0.mlir
         ${OUTPUT_DIR}/arg0.data
         ${OUTPUT_DIR}/labels.txt
  COMMAND ${PYTHON} buddy-pytorch-importer.py
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          buddy-pytorch-importer.py
          ${MODEL_DIR}/alexnet/output/alexnet/model.pth
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing original FP32 AlexNet for NVIDIA"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Compile the host entry point. Each command must succeed before the next runs.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/forward_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_linalg.mlir
             ${OUTPUT_DIR}/forward_gpu.mlir
             ${OUTPUT_DIR}/forward_gpu_llvm.mlir
             ${OUTPUT_DIR}/forward_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward.mlir
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
  COMMENT "Compiling original FP32 AlexNet host entry point"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower model computation to GPU kernels and embed the NVIDIA binary.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/subgraph0_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_gpu_linalg.mlir
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
            -canonicalize
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
            -pass-pipeline "builtin.module(func.func(gpu-async-region),canonicalize)"
            -o ${OUTPUT_DIR}/subgraph0_gpu_async.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_gpu_async.mlir
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=${NV_CUBIN_VERSION} cubin-features=+ptx71 cubin-format=fatbin"
            -o ${OUTPUT_DIR}/subgraph0_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_gpu_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/subgraph0_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0_gpu.ll -filetype=obj -relocation-model=pic -O2 -o ${OUTPUT_DIR}/subgraph0_gpu.o
  DEPENDS ${OUTPUT_DIR}/subgraph0.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling original FP32 AlexNet NVIDIA kernels"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Host library containing the compiled NVIDIA kernels.
# ===----------------------------------------------------------------------===
add_library(alexnet-nv-origin-model SHARED EXCLUDE_FROM_ALL ${OUTPUT_DIR}/subgraph0_gpu.o ${OUTPUT_DIR}/forward_gpu.o)
set_target_properties(alexnet-nv-origin-model
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             LINKER_LANGUAGE CXX
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_link_directories(alexnet-nv-origin-model
  PRIVATE ${BUDDY_SOURCE}/llvm/build/lib
)
target_link_libraries(alexnet-nv-origin-model
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
add_library(alexnet-nv-origin-runner SHARED EXCLUDE_FROM_ALL runner.cpp)
set_target_properties(alexnet-nv-origin-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_include_directories(alexnet-nv-origin-runner
  PRIVATE ${LIB_DIR}
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/runtime/include
          ${COMPILER_BUILD}/runtime/include
)
target_link_directories(alexnet-nv-origin-runner
  PRIVATE ${BUDDY_SOURCE}/llvm/build/lib
)
target_link_libraries(alexnet-nv-origin-runner
  PRIVATE dl
          opencv_core
          opencv_imgproc
          opencv_imgcodecs
          mlir_c_runner_utils
)

# ===----------------------------------------------------------------------===
# Package the RAX
#
# Contents:
# model library (libalexnet-nv-origin-model.so),
# runner library (libalexnet-nv-origin-runner.so),
# FP32 parameters (arg0.data),
# class labels (labels.txt),
# MLIR runtime libraries (libmlir_c_runner_utils.so, libmlir_runner_utils.so,
# libmlir_async_runtime.so, libmlir_cuda_runtime.so).
#
# Output: output/alexnet.rax embeds these files alongside RHAL metadata that
# identifies the runner, model library, parameters and runtime dependencies.
#
# ===----------------------------------------------------------------------===
configure_file(alexnet.rhal.mlir.in ${OUTPUT_DIR}/alexnet.rhal.mlir @ONLY)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/alexnet.rax
  COMMAND ${COMPILER_BUILD}/bin/rax-pack ${OUTPUT_DIR}/alexnet.rhal.mlir
          -o ${OUTPUT_DIR}/alexnet.rax
          --embed-payload
  DEPENDS ${OUTPUT_DIR}/alexnet.rhal.mlir
          alexnet-nv-origin-model
          alexnet-nv-origin-runner
          ${OUTPUT_DIR}/arg0.data
          ${COMPILER_BUILD}/bin/rax-pack
          ${OUTPUT_DIR}/labels.txt
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_c_runner_utils.so
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_runner_utils.so
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_async_runtime.so
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_cuda_runtime.so
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging original FP32 NVIDIA AlexNet as RAX"
  VERBATIM
)
add_custom_target(buddy-nv-alexnet-rax-origin-build DEPENDS ${OUTPUT_DIR}/alexnet.rax)
