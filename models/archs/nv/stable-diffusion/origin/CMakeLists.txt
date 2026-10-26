set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

add_custom_command(
  OUTPUT ${OUTPUT_DIR}/arg0_text_encoder.data
         ${OUTPUT_DIR}/arg1_text_encoder.data
         ${OUTPUT_DIR}/arg0_unet.data
         ${OUTPUT_DIR}/arg0_vae.data
         ${OUTPUT_DIR}/forward_text_encoder.mlir
         ${OUTPUT_DIR}/subgraph0_text_encoder.mlir
         ${OUTPUT_DIR}/forward_unet.mlir
         ${OUTPUT_DIR}/subgraph0_unet.mlir
         ${OUTPUT_DIR}/forward_vae.mlir
         ${OUTPUT_DIR}/subgraph0_vae.mlir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
  COMMAND ${PYTHON} buddy-pytorch-importer.py
            --checkpoint "${MODEL_DIR}/stable-diffusion/output/stable-diffusion"
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing stable-diffusion for nv"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Compile the host entry point. Each command must succeed before the next runs.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward_text_encoder_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_text_encoder_trace.mlir
             ${OUTPUT_DIR}/forward_text_encoder_linalg.mlir
             ${OUTPUT_DIR}/forward_text_encoder_gpu.mlir
             ${OUTPUT_DIR}/forward_text_encoder_gpu_llvm.mlir
             ${OUTPUT_DIR}/forward_text_encoder_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_text_encoder.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/forward_text_encoder_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_text_encoder_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/forward_text_encoder_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_text_encoder_linalg.mlir
            -pass-pipeline "builtin.module(func.func(buffer-hoisting),canonicalize,cse,expand-strided-metadata,func.func(convert-memcpy-to-gpu,gpu-async-region))"
            -o ${OUTPUT_DIR}/forward_text_encoder_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_text_encoder_gpu.mlir
            -llvm-request-c-wrappers
            -gpu-to-llvm
            -o ${OUTPUT_DIR}/forward_text_encoder_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/forward_text_encoder_gpu_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/forward_text_encoder_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward_text_encoder_gpu.ll -filetype=obj -relocation-model=pic -O2 -o forward_text_encoder_gpu.o
  DEPENDS ${OUTPUT_DIR}/forward_text_encoder.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling stable-diffusion host entry point"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower model computation to GPU kernels and embed the NVIDIA binary.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0_text_encoder_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_text_encoder_trace.mlir
             ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_text_encoder_bufferized.mlir
             ${OUTPUT_DIR}/subgraph0_text_encoder_copies.mlir
             ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_gpu.mlir
             ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_async.mlir
             ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_llvm.mlir
             ${OUTPUT_DIR}/subgraph0_text_encoder_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/subgraph0_text_encoder_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_linalg.mlir
            -convert-elementwise-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -o ${OUTPUT_DIR}/subgraph0_text_encoder_bufferized.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder_bufferized.mlir
            -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))"
            -o ${OUTPUT_DIR}/subgraph0_text_encoder_copies.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder_copies.mlir
            -convert-linalg-to-parallel-loops
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse
            -o ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_gpu.mlir
            -pass-pipeline "builtin.module(func.func(convert-memcpy-to-gpu{process-args=0},gpu-async-region),canonicalize)"
            -o ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_async.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_async.mlir
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=${NV_CUBIN_VERSION} cubin-features=+ptx71 cubin-format=fatbin"
            -o ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/subgraph0_text_encoder_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0_text_encoder_gpu.ll -filetype=obj -relocation-model=pic -O2 -o subgraph0_text_encoder_gpu.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_text_encoder.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling stable-diffusion NVIDIA kernels"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Compile the host entry point. Each command must succeed before the next runs.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward_unet_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_unet_trace.mlir
             ${OUTPUT_DIR}/forward_unet_linalg.mlir
             ${OUTPUT_DIR}/forward_unet_gpu.mlir
             ${OUTPUT_DIR}/forward_unet_gpu_llvm.mlir
             ${OUTPUT_DIR}/forward_unet_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_unet.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/forward_unet_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_unet_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/forward_unet_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_unet_linalg.mlir
            -pass-pipeline "builtin.module(func.func(buffer-hoisting),canonicalize,cse,expand-strided-metadata,func.func(convert-memcpy-to-gpu,gpu-async-region))"
            -o ${OUTPUT_DIR}/forward_unet_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_unet_gpu.mlir
            -llvm-request-c-wrappers
            -gpu-to-llvm
            -o ${OUTPUT_DIR}/forward_unet_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/forward_unet_gpu_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/forward_unet_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward_unet_gpu.ll -filetype=obj -relocation-model=pic -O2 -o forward_unet_gpu.o
  DEPENDS ${OUTPUT_DIR}/forward_unet.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling stable-diffusion host entry point"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower model computation to GPU kernels and embed the NVIDIA binary.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0_unet_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_unet_trace.mlir
             ${OUTPUT_DIR}/subgraph0_unet_gpu_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_unet_bufferized.mlir
             ${OUTPUT_DIR}/subgraph0_unet_copies.mlir
             ${OUTPUT_DIR}/subgraph0_unet_gpu_gpu.mlir
             ${OUTPUT_DIR}/subgraph0_unet_gpu_async.mlir
             ${OUTPUT_DIR}/subgraph0_unet_gpu_llvm.mlir
             ${OUTPUT_DIR}/subgraph0_unet_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_unet.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/subgraph0_unet_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_unet_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_unet_gpu_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_unet_gpu_linalg.mlir
            -convert-elementwise-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -o ${OUTPUT_DIR}/subgraph0_unet_bufferized.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_unet_bufferized.mlir
            -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))"
            -o ${OUTPUT_DIR}/subgraph0_unet_copies.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_unet_copies.mlir
            -convert-linalg-to-parallel-loops
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse
            -o ${OUTPUT_DIR}/subgraph0_unet_gpu_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_unet_gpu_gpu.mlir
            -pass-pipeline "builtin.module(func.func(convert-memcpy-to-gpu{process-args=0},gpu-async-region),canonicalize)"
            -o ${OUTPUT_DIR}/subgraph0_unet_gpu_async.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_unet_gpu_async.mlir
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=${NV_CUBIN_VERSION} cubin-features=+ptx71 cubin-format=fatbin"
            -o ${OUTPUT_DIR}/subgraph0_unet_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_unet_gpu_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/subgraph0_unet_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0_unet_gpu.ll -filetype=obj -relocation-model=pic -O2 -o subgraph0_unet_gpu.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_unet.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling stable-diffusion NVIDIA kernels"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Compile the host entry point. Each command must succeed before the next runs.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward_vae_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_vae_trace.mlir
             ${OUTPUT_DIR}/forward_vae_linalg.mlir
             ${OUTPUT_DIR}/forward_vae_gpu.mlir
             ${OUTPUT_DIR}/forward_vae_gpu_llvm.mlir
             ${OUTPUT_DIR}/forward_vae_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_vae.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/forward_vae_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_vae_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/forward_vae_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_vae_linalg.mlir
            -pass-pipeline "builtin.module(func.func(buffer-hoisting),canonicalize,cse,expand-strided-metadata,func.func(convert-memcpy-to-gpu,gpu-async-region))"
            -o ${OUTPUT_DIR}/forward_vae_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_vae_gpu.mlir
            -llvm-request-c-wrappers
            -gpu-to-llvm
            -o ${OUTPUT_DIR}/forward_vae_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/forward_vae_gpu_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/forward_vae_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward_vae_gpu.ll -filetype=obj -relocation-model=pic -O2 -o forward_vae_gpu.o
  DEPENDS ${OUTPUT_DIR}/forward_vae.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling stable-diffusion host entry point"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower model computation to GPU kernels and embed the NVIDIA binary.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0_vae_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_vae_trace.mlir
             ${OUTPUT_DIR}/subgraph0_vae_gpu_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_vae_bufferized.mlir
             ${OUTPUT_DIR}/subgraph0_vae_copies.mlir
             ${OUTPUT_DIR}/subgraph0_vae_gpu_gpu.mlir
             ${OUTPUT_DIR}/subgraph0_vae_gpu_async.mlir
             ${OUTPUT_DIR}/subgraph0_vae_gpu_llvm.mlir
             ${OUTPUT_DIR}/subgraph0_vae_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_vae.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/subgraph0_vae_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_vae_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_vae_gpu_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_vae_gpu_linalg.mlir
            -convert-elementwise-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -o ${OUTPUT_DIR}/subgraph0_vae_bufferized.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_vae_bufferized.mlir
            -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))"
            -o ${OUTPUT_DIR}/subgraph0_vae_copies.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_vae_copies.mlir
            -convert-linalg-to-parallel-loops
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse
            -o ${OUTPUT_DIR}/subgraph0_vae_gpu_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_vae_gpu_gpu.mlir
            -pass-pipeline "builtin.module(func.func(convert-memcpy-to-gpu{process-args=0},gpu-async-region),canonicalize)"
            -o ${OUTPUT_DIR}/subgraph0_vae_gpu_async.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_vae_gpu_async.mlir
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=${NV_CUBIN_VERSION} cubin-features=+ptx71 cubin-format=fatbin"
            -o ${OUTPUT_DIR}/subgraph0_vae_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_vae_gpu_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/subgraph0_vae_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0_vae_gpu.ll -filetype=obj -relocation-model=pic -O2 -o subgraph0_vae_gpu.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_vae.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling stable-diffusion NVIDIA kernels"
  VERBATIM
)

add_library(stable-diffusion-nv-origin-model SHARED EXCLUDE_FROM_ALL forward_text_encoder_gpu.o
  subgraph0_text_encoder_gpu.o forward_unet_gpu.o subgraph0_unet_gpu.o forward_vae_gpu.o subgraph0_vae_gpu.o
)
set_target_properties(stable-diffusion-nv-origin-model
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             LINKER_LANGUAGE CXX
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_link_directories(stable-diffusion-nv-origin-model
  PRIVATE ${BUDDY_SOURCE}/llvm/build/lib
)
target_link_libraries(stable-diffusion-nv-origin-model
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
add_library(stable-diffusion-nv-origin-runner SHARED EXCLUDE_FROM_ALL runner.cpp)
set_target_properties(stable-diffusion-nv-origin-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_include_directories(stable-diffusion-nv-origin-runner
  PRIVATE ${LIB_DIR}
          ${OUTPUT_DIR}
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/runtime/include
          ${COMPILER_BUILD}/runtime/include
)
target_link_directories(stable-diffusion-nv-origin-runner
  PRIVATE ${BUDDY_SOURCE}/llvm/build/lib
)
target_link_libraries(stable-diffusion-nv-origin-runner
  PRIVATE dl
          PNG::PNG
          ${COMPILER_BUILD}/lib/libBuddyLibDIP.a
          mlir_c_runner_utils
)

# ===----------------------------------------------------------------------===
# Package the RAX
#
# Contents:
# model library (libstable-diffusion-nv-origin-model.so),
# runner library (libstable-diffusion-nv-origin-runner.so),
# FP32 text encoder parameters (arg0_text_encoder.data),
# INT64 text positions (arg1_text_encoder.data),
# FP32 denoiser parameters (arg0_unet.data),
# FP32 VAE parameters (arg0_vae.data),
# token vocabulary (vocab.txt),
# MLIR runtime libraries (libmlir_c_runner_utils.so, libmlir_runner_utils.so,
# libmlir_async_runtime.so, libmlir_cuda_runtime.so).
#
# Output: output/stable-diffusion.rax embeds these files alongside RHAL metadata that
# identifies the runner, model library, parameters and runtime dependencies.
#
# ===----------------------------------------------------------------------===
configure_file(stable-diffusion.rhal.mlir.in ${OUTPUT_DIR}/stable-diffusion.rhal.mlir @ONLY)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/stable-diffusion.rax
  COMMAND ${COMPILER_BUILD}/bin/rax-pack ${OUTPUT_DIR}/stable-diffusion.rhal.mlir
          -o ${OUTPUT_DIR}/stable-diffusion.rax
          --embed-payload
  DEPENDS ${OUTPUT_DIR}/stable-diffusion.rhal.mlir
          stable-diffusion-nv-origin-model
          stable-diffusion-nv-origin-runner
          ${COMPILER_BUILD}/bin/rax-pack
          ${OUTPUT_DIR}/arg0_text_encoder.data
          ${OUTPUT_DIR}/arg1_text_encoder.data
          ${OUTPUT_DIR}/arg0_unet.data
          ${OUTPUT_DIR}/arg0_vae.data
          ${MODEL_DIR}/stable-diffusion/vocab.txt
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_c_runner_utils.so
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_runner_utils.so
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_async_runtime.so
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_cuda_runtime.so
  COMMENT "Packaging NVIDIA stable-diffusion as RAX"
  VERBATIM
)
add_custom_target(buddy-nv-stable-diffusion-rax-origin-build DEPENDS ${OUTPUT_DIR}/stable-diffusion.rax)
