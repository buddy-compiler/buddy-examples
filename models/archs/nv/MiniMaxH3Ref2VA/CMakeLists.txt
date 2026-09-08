set(MODEL_MINIMAX_H3_REF2VA_DIR ${MODEL_DIR}/MiniMaxH3Ref2VA)
set(INTERFACE_DIR ${NV_INTERFACE_DIR})


add_custom_command(
  OUTPUT forward_text_encoder_gpu.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_text_encoder.mlir
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
  DEPENDS ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_text_encoder.mlir
          ${BUDDY_BINARY_DIR}/buddy-opt
  COMMENT "Building NV forward_text_encoder_gpu.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph0_text_encoder_gpu.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_text_encoder.mlir
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
          -convert-strided-memref-copy-to-linalg
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
  DEPENDS ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_text_encoder.mlir
          ${BUDDY_BINARY_DIR}/buddy-opt
  COMMENT "Building NV subgraph0_text_encoder_gpu.o"
  VERBATIM)


add_custom_command(
  OUTPUT forward_transformer_gpu.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_transformer.mlir
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_transformer_gpu.o
  DEPENDS ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_transformer.mlir
          ${BUDDY_BINARY_DIR}/buddy-opt
  COMMENT "Building NV forward_transformer_gpu.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph0_transformer_gpu.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_transformer.mlir
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
          -convert-strided-memref-copy-to-linalg
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_transformer_gpu.o
  DEPENDS ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_transformer.mlir
          ${BUDDY_BINARY_DIR}/buddy-opt
  COMMENT "Building NV subgraph0_transformer_gpu.o"
  VERBATIM)


add_custom_command(
  OUTPUT forward_visual_vae_gpu.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_visual_vae.mlir
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_visual_vae_gpu.o
  DEPENDS ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_visual_vae.mlir
          ${BUDDY_BINARY_DIR}/buddy-opt
  COMMENT "Building NV forward_visual_vae_gpu.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph0_visual_vae_gpu.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_visual_vae.mlir
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
          -convert-strided-memref-copy-to-linalg
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_visual_vae_gpu.o
  DEPENDS ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_visual_vae.mlir
          ${BUDDY_BINARY_DIR}/buddy-opt
  COMMENT "Building NV subgraph0_visual_vae_gpu.o"
  VERBATIM)


add_custom_command(
  OUTPUT forward_visual_vae_encode_gpu.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_visual_vae_encode.mlir
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_visual_vae_encode_gpu.o
  DEPENDS ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_visual_vae_encode.mlir
          ${BUDDY_BINARY_DIR}/buddy-opt
  COMMENT "Building NV forward_visual_vae_encode_gpu.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph0_visual_vae_encode_gpu.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_visual_vae_encode.mlir
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
          -convert-strided-memref-copy-to-linalg
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_visual_vae_encode_gpu.o
  DEPENDS ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_visual_vae_encode.mlir
          ${BUDDY_BINARY_DIR}/buddy-opt
  COMMENT "Building NV subgraph0_visual_vae_encode_gpu.o"
  VERBATIM)


add_custom_command(
  OUTPUT forward_audio_vae_gpu.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_audio_vae.mlir
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_audio_vae_gpu.o
  DEPENDS ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_audio_vae.mlir
          ${BUDDY_BINARY_DIR}/buddy-opt
  COMMENT "Building NV forward_audio_vae_gpu.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph0_audio_vae_gpu.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_audio_vae.mlir
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
          -convert-strided-memref-copy-to-linalg
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_audio_vae_gpu.o
  DEPENDS ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_audio_vae.mlir
          ${BUDDY_BINARY_DIR}/buddy-opt
  COMMENT "Building NV subgraph0_audio_vae_gpu.o"
  VERBATIM)

add_library(NvH3Ref2VA_text_encoder STATIC subgraph0_text_encoder_gpu.o forward_text_encoder_gpu.o)
set_target_properties(NvH3Ref2VA_text_encoder PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(NvH3Ref2VA_text_encoder minimax-h3-ref2va-model-build)
add_library(NvH3Ref2VA_transformer STATIC subgraph0_transformer_gpu.o forward_transformer_gpu.o)
set_target_properties(NvH3Ref2VA_transformer PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(NvH3Ref2VA_transformer minimax-h3-ref2va-model-build)
add_library(NvH3Ref2VA_visual_vae STATIC subgraph0_visual_vae_gpu.o forward_visual_vae_gpu.o)
set_target_properties(NvH3Ref2VA_visual_vae PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(NvH3Ref2VA_visual_vae minimax-h3-ref2va-model-build)
add_library(NvH3Ref2VA_visual_vae_encode STATIC subgraph0_visual_vae_encode_gpu.o forward_visual_vae_encode_gpu.o)
set_target_properties(NvH3Ref2VA_visual_vae_encode PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(NvH3Ref2VA_visual_vae_encode minimax-h3-ref2va-model-build)
add_library(NvH3Ref2VA_audio_vae STATIC subgraph0_audio_vae_gpu.o forward_audio_vae_gpu.o)
set_target_properties(NvH3Ref2VA_audio_vae PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(NvH3Ref2VA_audio_vae minimax-h3-ref2va-model-build)

add_executable(buddy-nv-minimax-h3-ref2va-run ${MODEL_MINIMAX_H3_REF2VA_DIR}/buddy-minimax-h3-ref2va-main.cpp)
add_dependencies(buddy-nv-minimax-h3-ref2va-run NvH3Ref2VA_text_encoder NvH3Ref2VA_transformer NvH3Ref2VA_visual_vae NvH3Ref2VA_visual_vae_encode NvH3Ref2VA_audio_vae)
target_compile_definitions(buddy-nv-minimax-h3-ref2va-run PRIVATE
  MINIMAX_H3_REF2VA_EXAMPLE_PATH="${MODEL_MINIMAX_H3_REF2VA_DIR}/")
target_include_directories(buddy-nv-minimax-h3-ref2va-run PRIVATE ${INTERFACE_DIR})
target_link_directories(buddy-nv-minimax-h3-ref2va-run PRIVATE ${NV_LLVM_LIBRARY_DIR} ${BUDDY_MLIR_DIR}/build/lib)
target_link_libraries(buddy-nv-minimax-h3-ref2va-run
  NvH3Ref2VA_text_encoder
  NvH3Ref2VA_transformer
  NvH3Ref2VA_visual_vae
  NvH3Ref2VA_visual_vae_encode
  NvH3Ref2VA_audio_vae
  NvHostCRunnerUtils
  ${NV_MLIR_C_RUNNER_UTILS_LIB}
  ${NV_MLIR_RUNNER_UTILS_LIB}
  ${NV_MLIR_ASYNC_RUNTIME_LIB}
  ${NV_MLIR_CUDA_RUNTIME_LIB}
  dl
  pthread
)

add_custom_target(sync-nv-minimax-h3-ref2va-deps
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/trace/cycle
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_MINIMAX_H3_REF2VA_DIR}/arg0_text_encoder.data
          ${CMAKE_CURRENT_BINARY_DIR}/arg0_text_encoder.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_MINIMAX_H3_REF2VA_DIR}/arg0_transformer.data
          ${CMAKE_CURRENT_BINARY_DIR}/arg0_transformer.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_MINIMAX_H3_REF2VA_DIR}/arg0_visual_vae.data
          ${CMAKE_CURRENT_BINARY_DIR}/arg0_visual_vae.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_MINIMAX_H3_REF2VA_DIR}/arg0_audio_vae.data
          ${CMAKE_CURRENT_BINARY_DIR}/arg0_audio_vae.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_MINIMAX_H3_REF2VA_DIR}/h3_api.py
          ${CMAKE_CURRENT_BINARY_DIR}/h3_api.py
  DEPENDS
          ${MODEL_MINIMAX_H3_REF2VA_DIR}/arg0_text_encoder.data
          ${MODEL_MINIMAX_H3_REF2VA_DIR}/arg0_transformer.data
          ${MODEL_MINIMAX_H3_REF2VA_DIR}/arg0_visual_vae.data
          ${MODEL_MINIMAX_H3_REF2VA_DIR}/arg0_audio_vae.data
          ${MODEL_MINIMAX_H3_REF2VA_DIR}/h3_api.py
  COMMENT "Syncing NV MiniMax-H3 FL2VA runtime deps"
  VERBATIM)
add_dependencies(buddy-nv-minimax-h3-ref2va-run sync-nv-minimax-h3-ref2va-deps)

if(DEFINED OUTPUT_BIN_DIR)
  set(OUT_DIR ${OUTPUT_BIN_DIR}/src/ModelTest/e2e/models/archs/nv/MiniMaxH3Ref2VA)
  add_custom_command(TARGET buddy-nv-minimax-h3-ref2va-run POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}/trace/cycle
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:buddy-nv-minimax-h3-ref2va-run>
            ${OUT_DIR}/buddy-nv-minimax-h3-ref2va-run
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${MODEL_MINIMAX_H3_REF2VA_DIR}/arg0_text_encoder.data
            ${OUT_DIR}/arg0_text_encoder.data
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${MODEL_MINIMAX_H3_REF2VA_DIR}/arg0_transformer.data
            ${OUT_DIR}/arg0_transformer.data
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${MODEL_MINIMAX_H3_REF2VA_DIR}/arg0_visual_vae.data
            ${OUT_DIR}/arg0_visual_vae.data
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${MODEL_MINIMAX_H3_REF2VA_DIR}/arg0_audio_vae.data
            ${OUT_DIR}/arg0_audio_vae.data
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${MODEL_MINIMAX_H3_REF2VA_DIR}/h3_api.py
            ${OUT_DIR}/h3_api.py
    COMMENT "Copying NV MiniMax-H3 FL2VA executable and deps"
    VERBATIM)
endif()
