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
            --repo-root ${REPO_ROOT}
            --output-dir ${OUTPUT_DIR}
            --compiler-build ${COMPILER_BUILD}
            --checkpoint "borno1/stable_diffusion_2_base"
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing stable-diffusion for nv"
  VERBATIM
)

# Host x86 + NVIDIA GPU path (copied from nv/LeNet, adapted).

add_custom_command(
  OUTPUT forward_text_encoder_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_text_encoder_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_text_encoder.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -buffer-hoisting
            -canonicalize
            -cse
            -expand-strided-metadata
            -convert-memcpy-to-gpu
            -gpu-async-region |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers
            --gpu-to-llvm
            -o ${OUTPUT_DIR}/forward_text_encoder_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_text_encoder.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -buffer-hoisting
            -canonicalize
            -cse
            -expand-strided-metadata
            -convert-memcpy-to-gpu
            -gpu-async-region |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers
            --gpu-to-llvm |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-translate -mlir-to-llvmir |
          ${BUDDY_SOURCE}/llvm/build/bin/llvm-as |
          ${BUDDY_SOURCE}/llvm/build/bin/llc -filetype=obj -relocation-model=pic -O2 -o ${CMAKE_CURRENT_BINARY_DIR}/forward_text_encoder_gpu.o
  DEPENDS ${OUTPUT_DIR}/forward_text_encoder.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
  COMMENT "Building NV forward_text_encoder_gpu.o"
  VERBATIM
)

add_custom_command(
  OUTPUT subgraph0_text_encoder_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_gpu.mlir
             ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_nvvm.mlir
             ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -o ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse |
          ${COMPILER_BUILD}/bin/buddy-opt
            -convert-memcpy-to-gpu=process-args=0
            -gpu-async-region
            -canonicalize |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin"
            -o ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_nvvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse |
          ${COMPILER_BUILD}/bin/buddy-opt
            -convert-memcpy-to-gpu=process-args=0
            -gpu-async-region
            -canonicalize |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin"
            -o ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_text_encoder_gpu_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse |
          ${COMPILER_BUILD}/bin/buddy-opt
            -convert-memcpy-to-gpu=process-args=0
            -gpu-async-region
            -canonicalize |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin" |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-translate -mlir-to-llvmir |
          ${BUDDY_SOURCE}/llvm/build/bin/llvm-as |
          ${BUDDY_SOURCE}/llvm/build/bin/llc -filetype=obj -relocation-model=pic -O2 -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_text_encoder_gpu.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_text_encoder.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
  COMMENT "Building NV subgraph0_text_encoder_gpu.o (TOSA→NVVM)"
  VERBATIM
)

add_custom_command(
  OUTPUT forward_unet_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_unet_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_unet.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -buffer-hoisting
            -canonicalize
            -cse
            -expand-strided-metadata
            -convert-memcpy-to-gpu
            -gpu-async-region |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers
            --gpu-to-llvm
            -o ${OUTPUT_DIR}/forward_unet_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_unet.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -buffer-hoisting
            -canonicalize
            -cse
            -expand-strided-metadata
            -convert-memcpy-to-gpu
            -gpu-async-region |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers
            --gpu-to-llvm |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-translate -mlir-to-llvmir |
          ${BUDDY_SOURCE}/llvm/build/bin/llvm-as |
          ${BUDDY_SOURCE}/llvm/build/bin/llc -filetype=obj -relocation-model=pic -O2 -o ${CMAKE_CURRENT_BINARY_DIR}/forward_unet_gpu.o
  DEPENDS ${OUTPUT_DIR}/forward_unet.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
  COMMENT "Building NV forward_unet_gpu.o"
  VERBATIM
)

add_custom_command(
  OUTPUT subgraph0_unet_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_unet_gpu_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_unet_gpu_gpu.mlir
             ${OUTPUT_DIR}/subgraph0_unet_gpu_nvvm.mlir
             ${OUTPUT_DIR}/subgraph0_unet_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_unet.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -o ${OUTPUT_DIR}/subgraph0_unet_gpu_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_unet.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse |
          ${COMPILER_BUILD}/bin/buddy-opt
            -convert-memcpy-to-gpu=process-args=0
            -gpu-async-region
            -canonicalize |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin"
            -o ${OUTPUT_DIR}/subgraph0_unet_gpu_nvvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_unet.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse |
          ${COMPILER_BUILD}/bin/buddy-opt
            -convert-memcpy-to-gpu=process-args=0
            -gpu-async-region
            -canonicalize |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin"
            -o ${OUTPUT_DIR}/subgraph0_unet_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_unet.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_unet_gpu_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_unet.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse |
          ${COMPILER_BUILD}/bin/buddy-opt
            -convert-memcpy-to-gpu=process-args=0
            -gpu-async-region
            -canonicalize |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin" |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-translate -mlir-to-llvmir |
          ${BUDDY_SOURCE}/llvm/build/bin/llvm-as |
          ${BUDDY_SOURCE}/llvm/build/bin/llc -filetype=obj -relocation-model=pic -O2 -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_unet_gpu.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_unet.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
  COMMENT "Building NV subgraph0_unet_gpu.o (TOSA→NVVM)"
  VERBATIM
)

add_custom_command(
  OUTPUT forward_vae_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_vae_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_vae.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -buffer-hoisting
            -canonicalize
            -cse
            -expand-strided-metadata
            -convert-memcpy-to-gpu
            -gpu-async-region |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers
            --gpu-to-llvm
            -o ${OUTPUT_DIR}/forward_vae_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_vae.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -buffer-hoisting
            -canonicalize
            -cse
            -expand-strided-metadata
            -convert-memcpy-to-gpu
            -gpu-async-region |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers
            --gpu-to-llvm |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-translate -mlir-to-llvmir |
          ${BUDDY_SOURCE}/llvm/build/bin/llvm-as |
          ${BUDDY_SOURCE}/llvm/build/bin/llc -filetype=obj -relocation-model=pic -O2 -o ${CMAKE_CURRENT_BINARY_DIR}/forward_vae_gpu.o
  DEPENDS ${OUTPUT_DIR}/forward_vae.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
  COMMENT "Building NV forward_vae_gpu.o"
  VERBATIM
)

add_custom_command(
  OUTPUT subgraph0_vae_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_vae_gpu_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_vae_gpu_gpu.mlir
             ${OUTPUT_DIR}/subgraph0_vae_gpu_nvvm.mlir
             ${OUTPUT_DIR}/subgraph0_vae_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_vae.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -o ${OUTPUT_DIR}/subgraph0_vae_gpu_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_vae.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse |
          ${COMPILER_BUILD}/bin/buddy-opt
            -convert-memcpy-to-gpu=process-args=0
            -gpu-async-region
            -canonicalize |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin"
            -o ${OUTPUT_DIR}/subgraph0_vae_gpu_nvvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_vae.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse |
          ${COMPILER_BUILD}/bin/buddy-opt
            -convert-memcpy-to-gpu=process-args=0
            -gpu-async-region
            -canonicalize |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin"
            -o ${OUTPUT_DIR}/subgraph0_vae_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_vae.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_vae_gpu_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_vae.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse |
          ${COMPILER_BUILD}/bin/buddy-opt
            -convert-memcpy-to-gpu=process-args=0
            -gpu-async-region
            -canonicalize |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin" |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-translate -mlir-to-llvmir |
          ${BUDDY_SOURCE}/llvm/build/bin/llvm-as |
          ${BUDDY_SOURCE}/llvm/build/bin/llc -filetype=obj -relocation-model=pic -O2 -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_vae_gpu.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_vae.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
  COMMENT "Building NV subgraph0_vae_gpu.o (TOSA→NVVM)"
  VERBATIM
)

add_library(stable-diffusion-nv-textencoder-model STATIC EXCLUDE_FROM_ALL subgraph0_text_encoder_gpu.o
  forward_text_encoder_gpu.o
)
set_target_properties(stable-diffusion-nv-textencoder-model PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
add_library(stable-diffusion-nv-unet-model STATIC EXCLUDE_FROM_ALL subgraph0_unet_gpu.o forward_unet_gpu.o)
set_target_properties(stable-diffusion-nv-unet-model PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
add_library(stable-diffusion-nv-vae-model STATIC EXCLUDE_FROM_ALL subgraph0_vae_gpu.o forward_vae_gpu.o)
set_target_properties(stable-diffusion-nv-vae-model PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
set_target_properties(stable-diffusion-nv-textencoder-model stable-diffusion-nv-unet-model
  stable-diffusion-nv-vae-model PROPERTIES LINKER_LANGUAGE CXX
)

add_executable(buddy-nv-stable-diffusion-run EXCLUDE_FROM_ALL ${MODEL_DIR}/stable-diffusion/main.cpp)
set_target_properties(buddy-nv-stable-diffusion-run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
add_dependencies(
  buddy-nv-stable-diffusion-run
  stable-diffusion-nv-textencoder-model
  stable-diffusion-nv-unet-model
  stable-diffusion-nv-vae-model
)
target_include_directories(buddy-nv-stable-diffusion-run PRIVATE ${BUDDY_SOURCE}/frontend/Interfaces
                                                                 ${MODEL_DIR}/stable-diffusion/include
)
target_link_directories(buddy-nv-stable-diffusion-run PRIVATE ${BUDDY_SOURCE}/llvm/build/lib ${BUDDY_SOURCE}/build/lib)
target_link_libraries(buddy-nv-stable-diffusion-run stable-diffusion-nv-textencoder-model stable-diffusion-nv-unet-model
                                                                                          stable-diffusion-nv-vae-model
                                                                                          nv-runtime
                                                                                          BuddyLibDIP
                                                                                          PNG::PNG
                                                                                          mlir_c_runner_utils
                                                                                          mlir_runner_utils
                                                                                          mlir_async_runtime
                                                                                          mlir_cuda_runtime
                                                                                          dl
                                                                                          pthread
)

add_custom_target(
  stable-diffusion-model-build
  DEPENDS stable-diffusion-nv-textencoder-model stable-diffusion-nv-unet-model stable-diffusion-nv-vae-model
)
