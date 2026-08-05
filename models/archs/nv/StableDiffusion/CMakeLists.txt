# Host x86 + NVIDIA GPU path (copied from nv/LeNet, adapted).

set(MODEL_STABLEDIFFUSION_DIR ${MODEL_DIR}/StableDiffusion)
set(INTERFACE_DIR ${NV_INTERFACE_DIR})
set(INCLUDE_DIR ${MODEL_STABLEDIFFUSION_DIR}/include)


add_custom_command(
  OUTPUT forward_text_encoder_gpu.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/forward_text_encoder.mlir
          -convert-trace-to-llvm=cycle-trace |
          ${BUDDY_BINARY_DIR}/buddy-opt
          -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
          -buffer-hoisting
          -canonicalize -cse -expand-strided-metadata
          -convert-memcpy-to-gpu -gpu-async-region |
          ${NV_LLVM_TOOLS_BINARY_DIR}/mlir-opt
          -llvm-request-c-wrappers --gpu-to-llvm |
          ${NV_LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${NV_LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${NV_LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O0
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_text_encoder_gpu.o
  DEPENDS ${MODEL_STABLEDIFFUSION_DIR}/forward_text_encoder.mlir
          ${BUDDY_BINARY_DIR}/buddy-opt
  COMMENT "Building NV forward_text_encoder_gpu.o"
  VERBATIM)


add_custom_command(
  OUTPUT subgraph0_text_encoder_gpu.o
  BYPRODUCTS subgraph0_text_encoder_gpu_linalg.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_text_encoder.mlir
          -convert-trace-to-llvm=cycle-trace |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))"
            > ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_text_encoder_gpu_linalg.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_text_encoder.mlir
          -convert-trace-to-llvm=cycle-trace |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
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
          ${BUDDY_BINARY_DIR}/buddy-opt
          -convert-memcpy-to-gpu=${NV_CONVERT_MEMCPY_TO_GPU_OPTION_DISABLE_PROCESS_ARG}
          -gpu-async-region -canonicalize |
          ${NV_LLVM_TOOLS_BINARY_DIR}/mlir-opt
          -llvm-request-c-wrappers
          -gpu-lower-to-nvvm-pipeline=${NV_LOWER_TO_NVVM_OPTION} |
          ${NV_LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${NV_LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${NV_LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O0
          -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_text_encoder_gpu.o
  DEPENDS ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_text_encoder.mlir
          ${BUDDY_BINARY_DIR}/buddy-opt
  COMMENT "Building NV subgraph0_text_encoder_gpu.o (TOSA→NVVM)"
  VERBATIM)


add_custom_command(
  OUTPUT forward_unet_gpu.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/forward_unet.mlir
          -convert-trace-to-llvm=cycle-trace |
          ${BUDDY_BINARY_DIR}/buddy-opt
          -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
          -buffer-hoisting
          -canonicalize -cse -expand-strided-metadata
          -convert-memcpy-to-gpu -gpu-async-region |
          ${NV_LLVM_TOOLS_BINARY_DIR}/mlir-opt
          -llvm-request-c-wrappers --gpu-to-llvm |
          ${NV_LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${NV_LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${NV_LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O0
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_unet_gpu.o
  DEPENDS ${MODEL_STABLEDIFFUSION_DIR}/forward_unet.mlir
          ${BUDDY_BINARY_DIR}/buddy-opt
  COMMENT "Building NV forward_unet_gpu.o"
  VERBATIM)


add_custom_command(
  OUTPUT subgraph0_unet_gpu.o
  BYPRODUCTS subgraph0_unet_gpu_linalg.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_unet.mlir
          -convert-trace-to-llvm=cycle-trace |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))"
            > ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_unet_gpu_linalg.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_unet.mlir
          -convert-trace-to-llvm=cycle-trace |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
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
          ${BUDDY_BINARY_DIR}/buddy-opt
          -convert-memcpy-to-gpu=${NV_CONVERT_MEMCPY_TO_GPU_OPTION_DISABLE_PROCESS_ARG}
          -gpu-async-region -canonicalize |
          ${NV_LLVM_TOOLS_BINARY_DIR}/mlir-opt
          -llvm-request-c-wrappers
          -gpu-lower-to-nvvm-pipeline=${NV_LOWER_TO_NVVM_OPTION} |
          ${NV_LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${NV_LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${NV_LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O0
          -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_unet_gpu.o
  DEPENDS ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_unet.mlir
          ${BUDDY_BINARY_DIR}/buddy-opt
  COMMENT "Building NV subgraph0_unet_gpu.o (TOSA→NVVM)"
  VERBATIM)


add_custom_command(
  OUTPUT forward_vae_gpu.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/forward_vae.mlir
          -convert-trace-to-llvm=cycle-trace |
          ${BUDDY_BINARY_DIR}/buddy-opt
          -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
          -buffer-hoisting
          -canonicalize -cse -expand-strided-metadata
          -convert-memcpy-to-gpu -gpu-async-region |
          ${NV_LLVM_TOOLS_BINARY_DIR}/mlir-opt
          -llvm-request-c-wrappers --gpu-to-llvm |
          ${NV_LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${NV_LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${NV_LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O0
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_vae_gpu.o
  DEPENDS ${MODEL_STABLEDIFFUSION_DIR}/forward_vae.mlir
          ${BUDDY_BINARY_DIR}/buddy-opt
  COMMENT "Building NV forward_vae_gpu.o"
  VERBATIM)


add_custom_command(
  OUTPUT subgraph0_vae_gpu.o
  BYPRODUCTS subgraph0_vae_gpu_linalg.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_vae.mlir
          -convert-trace-to-llvm=cycle-trace |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))"
            > ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_vae_gpu_linalg.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_vae.mlir
          -convert-trace-to-llvm=cycle-trace |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
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
          ${BUDDY_BINARY_DIR}/buddy-opt
          -convert-memcpy-to-gpu=${NV_CONVERT_MEMCPY_TO_GPU_OPTION_DISABLE_PROCESS_ARG}
          -gpu-async-region -canonicalize |
          ${NV_LLVM_TOOLS_BINARY_DIR}/mlir-opt
          -llvm-request-c-wrappers
          -gpu-lower-to-nvvm-pipeline=${NV_LOWER_TO_NVVM_OPTION} |
          ${NV_LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${NV_LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${NV_LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O0
          -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_vae_gpu.o
  DEPENDS ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_vae.mlir
          ${BUDDY_BINARY_DIR}/buddy-opt
  COMMENT "Building NV subgraph0_vae_gpu.o (TOSA→NVVM)"
  VERBATIM)


add_library(NvTEXTENCODER STATIC subgraph0_text_encoder_gpu.o forward_text_encoder_gpu.o)
add_library(NvUNET STATIC subgraph0_unet_gpu.o forward_unet_gpu.o)
add_library(NvVAE STATIC subgraph0_vae_gpu.o forward_vae_gpu.o)
set_target_properties(NvTEXTENCODER NvUNET NvVAE PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(NvTEXTENCODER stable-diffusion-model-build)
add_dependencies(NvUNET stable-diffusion-model-build)
add_dependencies(NvVAE stable-diffusion-model-build)

add_executable(buddy-nv-stable-diffusion-run ${MODEL_STABLEDIFFUSION_DIR}/buddy-stable-diffusion-main.cpp)
add_dependencies(buddy-nv-stable-diffusion-run NvTEXTENCODER NvUNET NvVAE)
target_include_directories(buddy-nv-stable-diffusion-run PRIVATE ${INTERFACE_DIR} ${INCLUDE_DIR})
target_link_directories(buddy-nv-stable-diffusion-run PRIVATE ${NV_LLVM_LIBRARY_DIR} ${BUDDY_MLIR_DIR}/build/lib)
target_link_libraries(buddy-nv-stable-diffusion-run
  NvTEXTENCODER
  NvUNET
  NvVAE
  NvHostCRunnerUtils
  ${NV_BUDDY_LIB_DIP}
  PNG::PNG
  ${NV_MLIR_C_RUNNER_UTILS_LIB}
  ${NV_MLIR_RUNNER_UTILS_LIB}
  ${NV_MLIR_ASYNC_RUNTIME_LIB}
  ${NV_MLIR_CUDA_RUNTIME_LIB}
  dl
  pthread
)

add_custom_target(sync-nv-stable-diffusion-deps
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/trace/cycle
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_STABLEDIFFUSION_DIR}/arg0_text_encoder.data
          ${CMAKE_CURRENT_BINARY_DIR}/arg0_text_encoder.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_STABLEDIFFUSION_DIR}/arg1_text_encoder.data
          ${CMAKE_CURRENT_BINARY_DIR}/arg1_text_encoder.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_STABLEDIFFUSION_DIR}/arg0_unet.data
          ${CMAKE_CURRENT_BINARY_DIR}/arg0_unet.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_STABLEDIFFUSION_DIR}/arg0_vae.data
          ${CMAKE_CURRENT_BINARY_DIR}/arg0_vae.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_STABLEDIFFUSION_DIR}/vocab.txt
          ${CMAKE_CURRENT_BINARY_DIR}/vocab.txt
  DEPENDS
          ${MODEL_STABLEDIFFUSION_DIR}/arg0_text_encoder.data
          ${MODEL_STABLEDIFFUSION_DIR}/arg1_text_encoder.data
          ${MODEL_STABLEDIFFUSION_DIR}/arg0_unet.data
          ${MODEL_STABLEDIFFUSION_DIR}/arg0_vae.data
          ${MODEL_STABLEDIFFUSION_DIR}/vocab.txt
  COMMENT "Syncing NV stable-diffusion runtime deps"
  VERBATIM)
add_dependencies(buddy-nv-stable-diffusion-run sync-nv-stable-diffusion-deps)

if(DEFINED OUTPUT_BIN_DIR)
  set(OUT_DIR ${OUTPUT_BIN_DIR}/src/ModelTest/e2e/models/archs/nv/StableDiffusion)
  add_custom_command(TARGET buddy-nv-stable-diffusion-run POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}/trace/cycle
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:buddy-nv-stable-diffusion-run>
            ${OUT_DIR}/buddy-nv-stable-diffusion-run
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${MODEL_STABLEDIFFUSION_DIR}/arg0_text_encoder.data
            ${OUT_DIR}/arg0_text_encoder.data
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${MODEL_STABLEDIFFUSION_DIR}/arg1_text_encoder.data
            ${OUT_DIR}/arg1_text_encoder.data
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${MODEL_STABLEDIFFUSION_DIR}/arg0_unet.data
            ${OUT_DIR}/arg0_unet.data
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${MODEL_STABLEDIFFUSION_DIR}/arg0_vae.data
            ${OUT_DIR}/arg0_vae.data
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${MODEL_STABLEDIFFUSION_DIR}/vocab.txt
            ${OUT_DIR}/vocab.txt
    COMMENT "Copying NV stable-diffusion executable and deps to output directory"
    VERBATIM)
endif()

add_custom_target(buddy-nv-stable-diffusion-all-run DEPENDS buddy-nv-stable-diffusion-run)
