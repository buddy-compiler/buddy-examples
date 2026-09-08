# Host x86 + NVIDIA GPU path (copied from nv/LeNet, adapted).

set(MODEL_QWEN3_DIR ${MODEL_DIR}/Qwen3)
set(INTERFACE_DIR ${NV_INTERFACE_DIR})
set(INCLUDE_DIR ${MODEL_QWEN3_DIR})


add_custom_command(
  OUTPUT forward_prefill_gpu.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_QWEN3_DIR}/forward_prefill_0_6b.mlir
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill_gpu.o
  DEPENDS ${MODEL_QWEN3_DIR}/forward_prefill_0_6b.mlir
          ${BUDDY_BINARY_DIR}/buddy-opt
  COMMENT "Building NV forward_prefill_gpu.o"
  VERBATIM)


add_custom_command(
  OUTPUT subgraph0_prefill_gpu.o
  BYPRODUCTS subgraph0_prefill_gpu_linalg.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_QWEN3_DIR}/subgraph0_prefill_0_6b.mlir
          -convert-trace-to-llvm=cycle-trace |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))"
            > ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill_gpu_linalg.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_QWEN3_DIR}/subgraph0_prefill_0_6b.mlir
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill_gpu.o
  DEPENDS ${MODEL_QWEN3_DIR}/subgraph0_prefill_0_6b.mlir
          ${BUDDY_BINARY_DIR}/buddy-opt
  COMMENT "Building NV subgraph0_prefill_gpu.o (TOSA→NVVM)"
  VERBATIM)


add_custom_command(
  OUTPUT forward_decode_gpu.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_QWEN3_DIR}/forward_decode_0_6b.mlir
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_decode_gpu.o
  DEPENDS ${MODEL_QWEN3_DIR}/forward_decode_0_6b.mlir
          ${BUDDY_BINARY_DIR}/buddy-opt
  COMMENT "Building NV forward_decode_gpu.o"
  VERBATIM)


add_custom_command(
  OUTPUT subgraph0_decode_gpu.o
  BYPRODUCTS subgraph0_decode_gpu_linalg.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_QWEN3_DIR}/subgraph0_decode_0_6b.mlir
          -convert-trace-to-llvm=cycle-trace |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))"
            > ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode_gpu_linalg.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_QWEN3_DIR}/subgraph0_decode_0_6b.mlir
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode_gpu.o
  DEPENDS ${MODEL_QWEN3_DIR}/subgraph0_decode_0_6b.mlir
          ${BUDDY_BINARY_DIR}/buddy-opt
  COMMENT "Building NV subgraph0_decode_gpu.o (TOSA→NVVM)"
  VERBATIM)


add_library(NvQWEN3 STATIC subgraph0_prefill_gpu.o forward_prefill_gpu.o subgraph0_decode_gpu.o forward_decode_gpu.o)
set_target_properties(NvQWEN3 PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(NvQWEN3 qwen3-model-build)

add_executable(buddy-nv-qwen3-run ${MODEL_QWEN3_DIR}/buddy-qwen3-0.6b-main.cpp)
add_dependencies(buddy-nv-qwen3-run NvQWEN3)
target_include_directories(buddy-nv-qwen3-run PRIVATE ${INTERFACE_DIR} ${INCLUDE_DIR})
target_link_directories(buddy-nv-qwen3-run PRIVATE ${NV_LLVM_LIBRARY_DIR} ${BUDDY_MLIR_DIR}/build/lib)
target_compile_definitions(buddy-nv-qwen3-run PRIVATE
  QWEN3_0_6B_EXAMPLE_PATH="${MODEL_QWEN3_DIR}/"
)
target_link_libraries(buddy-nv-qwen3-run
  NvQWEN3
  NvHostCRunnerUtils
  ${NV_MLIR_C_RUNNER_UTILS_LIB}
  ${NV_MLIR_RUNNER_UTILS_LIB}
  ${NV_MLIR_ASYNC_RUNTIME_LIB}
  ${NV_MLIR_CUDA_RUNTIME_LIB}
  dl
  pthread
)

add_custom_target(sync-nv-qwen3-deps
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/trace/cycle
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_QWEN3_DIR}/arg0_0_6b.data
          ${CMAKE_CURRENT_BINARY_DIR}/arg0_0_6b.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_QWEN3_DIR}/arg1_0_6b.data
          ${CMAKE_CURRENT_BINARY_DIR}/arg1_0_6b.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_QWEN3_DIR}/vocab.txt
          ${CMAKE_CURRENT_BINARY_DIR}/vocab.txt
  COMMAND bash -c "if [ -d \"$0\" ]; then mkdir -p \"$1\" && cp -rf \"$0/.\" \"$1/\"; fi"
          ${MODEL_QWEN3_DIR}/quant/out
          ${CMAKE_CURRENT_BINARY_DIR}/quant/out
  DEPENDS
          ${MODEL_QWEN3_DIR}/arg0_0_6b.data
          ${MODEL_QWEN3_DIR}/arg1_0_6b.data
          ${MODEL_QWEN3_DIR}/vocab.txt
  COMMENT "Syncing NV qwen3 runtime deps"
  VERBATIM)
add_dependencies(buddy-nv-qwen3-run sync-nv-qwen3-deps)

if(DEFINED OUTPUT_BIN_DIR)
  set(OUT_DIR ${OUTPUT_BIN_DIR}/src/ModelTest/e2e/models/archs/nv/Qwen3)
  add_custom_command(TARGET buddy-nv-qwen3-run POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}/trace/cycle
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:buddy-nv-qwen3-run>
            ${OUT_DIR}/buddy-nv-qwen3-run
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${MODEL_QWEN3_DIR}/arg0_0_6b.data
            ${OUT_DIR}/arg0_0_6b.data
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${MODEL_QWEN3_DIR}/arg1_0_6b.data
            ${OUT_DIR}/arg1_0_6b.data
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${MODEL_QWEN3_DIR}/vocab.txt
            ${OUT_DIR}/vocab.txt
    COMMAND bash -c "if [ -d \"$0\" ]; then mkdir -p \"$1\" && cp -rf \"$0/.\" \"$1/\"; fi"
            ${MODEL_QWEN3_DIR}/quant/out
            ${OUT_DIR}/quant/out
    COMMENT "Copying NV qwen3 executable and deps to output directory"
    VERBATIM)
endif()

add_custom_target(buddy-nv-qwen3-all-run DEPENDS buddy-nv-qwen3-run)
