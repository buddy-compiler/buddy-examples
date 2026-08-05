# Host x86 + NVIDIA GPU path (copied from nv/LeNet, adapted).

set(MODEL_MOBILENETV3_DIR ${MODEL_DIR}/MobileNetV3)
set(INTERFACE_DIR ${NV_INTERFACE_DIR})
set(INCLUDE_DIR ${MODELTEST_LIB_DIR})


add_custom_command(
  OUTPUT forward_gpu.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_MOBILENETV3_DIR}/forward.mlir
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_gpu.o
  DEPENDS ${MODEL_MOBILENETV3_DIR}/forward.mlir
          ${BUDDY_BINARY_DIR}/buddy-opt
  COMMENT "Building NV forward_gpu.o"
  VERBATIM)


add_custom_command(
  OUTPUT subgraph0_gpu.o
  BYPRODUCTS subgraph0_gpu_linalg.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_MOBILENETV3_DIR}/subgraph0.mlir
          -convert-trace-to-llvm=cycle-trace |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))"
            > ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_gpu_linalg.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_MOBILENETV3_DIR}/subgraph0.mlir
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_gpu.o
  DEPENDS ${MODEL_MOBILENETV3_DIR}/subgraph0.mlir
          ${BUDDY_BINARY_DIR}/buddy-opt
  COMMENT "Building NV subgraph0_gpu.o (TOSA→NVVM)"
  VERBATIM)

add_library(NvMOBILENETV3 STATIC subgraph0_gpu.o forward_gpu.o)
set_target_properties(NvMOBILENETV3 PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(NvMOBILENETV3 mobilenetv3-model-build)

add_executable(buddy-nv-mobilenetv3-run ${MODEL_MOBILENETV3_DIR}/buddy-mobilenetv3-main.cpp)
add_dependencies(buddy-nv-mobilenetv3-run NvMOBILENETV3)
target_include_directories(buddy-nv-mobilenetv3-run PRIVATE ${INTERFACE_DIR} ${INCLUDE_DIR})
target_link_directories(buddy-nv-mobilenetv3-run PRIVATE ${NV_LLVM_LIBRARY_DIR} ${BUDDY_MLIR_DIR}/build/lib)
target_link_libraries(buddy-nv-mobilenetv3-run
  NvMOBILENETV3
  NvHostCRunnerUtils
  PNG::PNG
  ${NV_BUDDY_LIB_DIP}
  ${NV_MLIR_C_RUNNER_UTILS_LIB}
  ${NV_MLIR_RUNNER_UTILS_LIB}
  ${NV_MLIR_ASYNC_RUNTIME_LIB}
  ${NV_MLIR_CUDA_RUNTIME_LIB}
  dl
  pthread
)

add_custom_target(sync-nv-mobilenetv3-deps
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/trace/cycle
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${MODEL_MOBILENETV3_DIR}/images
          ${CMAKE_CURRENT_BINARY_DIR}/images
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_MOBILENETV3_DIR}/arg0.data
          ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_MOBILENETV3_DIR}/Labels.txt
          ${CMAKE_CURRENT_BINARY_DIR}/Labels.txt
  DEPENDS
          ${MODEL_MOBILENETV3_DIR}/arg0.data
          ${MODEL_MOBILENETV3_DIR}/Labels.txt
  COMMENT "Syncing NV mobilenetv3 runtime deps"
  VERBATIM)
add_dependencies(buddy-nv-mobilenetv3-run sync-nv-mobilenetv3-deps)

if(DEFINED OUTPUT_BIN_DIR)
  set(OUT_DIR ${OUTPUT_BIN_DIR}/src/ModelTest/e2e/models/archs/nv/MobileNetV3)
  add_custom_command(TARGET buddy-nv-mobilenetv3-run POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}/trace/cycle
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:buddy-nv-mobilenetv3-run>
            ${OUT_DIR}/buddy-nv-mobilenetv3-run
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${MODEL_MOBILENETV3_DIR}/images
            ${OUT_DIR}/images
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${MODEL_MOBILENETV3_DIR}/arg0.data
            ${OUT_DIR}/arg0.data
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${MODEL_MOBILENETV3_DIR}/Labels.txt
            ${OUT_DIR}/Labels.txt
    COMMENT "Copying NV mobilenetv3 executable and deps to output directory"
    VERBATIM)
endif()

add_custom_target(buddy-nv-mobilenetv3-all-run DEPENDS buddy-nv-mobilenetv3-run)
