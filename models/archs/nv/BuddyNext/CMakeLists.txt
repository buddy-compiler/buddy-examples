# Host x86 + NVIDIA GPU path (copied from nv/LeNet, adapted).

set(MODEL_BUDDYNEXT_DIR ${MODEL_DIR}/BuddyNext)
set(TOSA_PIPELINE "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))")

file(GLOB BUDDYNEXT_PREFILL_MLIR_FILES CONFIGURE_DEPENDS
  ${MODEL_BUDDYNEXT_DIR}/prefill/*.mlir
)
file(GLOB BUDDYNEXT_DECODE_MLIR_FILES CONFIGURE_DEPENDS
  ${MODEL_BUDDYNEXT_DIR}/decode/*.mlir
)

function(add_buddynext_nv_target target_name mlir_file object_name)
  add_custom_command(
    OUTPUT ${object_name}
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${mlir_file}
              -convert-trace-to-llvm=cycle-trace |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -pass-pipeline ${TOSA_PIPELINE} |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -eliminate-empty-tensors
              -empty-tensor-to-alloc-tensor
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
              -o ${CMAKE_CURRENT_BINARY_DIR}/${object_name}
    DEPENDS ${mlir_file} ${BUDDY_BINARY_DIR}/buddy-opt
    COMMENT "Building NV ${target_name} object"
    VERBATIM
  )

  add_library(${target_name} STATIC ${object_name})
  set_target_properties(${target_name} PROPERTIES LINKER_LANGUAGE CXX)
  add_dependencies(${target_name} buddynext-model-build)
endfunction()

set(BUDDYNEXT_PREFILL_TARGETS)
foreach(mlir_file ${BUDDYNEXT_PREFILL_MLIR_FILES})
  get_filename_component(mlir_base ${mlir_file} NAME_WE)
  string(REPLACE "-" "_" mlir_base_sanitized ${mlir_base})
  set(lib_target NvBUDDYNEXT_prefill_${mlir_base_sanitized})
  set(obj_file prefill_${mlir_base_sanitized}_gpu.o)
  add_buddynext_nv_target(${lib_target} ${mlir_file} ${obj_file})

  set(run_target buddy-nv-buddynext-prefill-${mlir_base}-run)
  add_executable(${run_target} ${MODEL_BUDDYNEXT_DIR}/buddy-next-runtime.cpp)
  add_dependencies(${run_target} ${lib_target})
  target_link_directories(${run_target} PRIVATE ${NV_LLVM_LIBRARY_DIR})
  target_link_libraries(${run_target}
    ${lib_target}
    NvHostCRunnerUtils
    ${NV_MLIR_C_RUNNER_UTILS_LIB}
    ${NV_MLIR_RUNNER_UTILS_LIB}
    ${NV_MLIR_ASYNC_RUNTIME_LIB}
    ${NV_MLIR_CUDA_RUNTIME_LIB}
    dl
    pthread
  )
  list(APPEND BUDDYNEXT_PREFILL_TARGETS ${run_target})
endforeach()

set(BUDDYNEXT_DECODE_TARGETS)
foreach(mlir_file ${BUDDYNEXT_DECODE_MLIR_FILES})
  get_filename_component(mlir_base ${mlir_file} NAME_WE)
  string(REPLACE "-" "_" mlir_base_sanitized ${mlir_base})
  set(lib_target NvBUDDYNEXT_decode_${mlir_base_sanitized})
  set(obj_file decode_${mlir_base_sanitized}_gpu.o)
  add_buddynext_nv_target(${lib_target} ${mlir_file} ${obj_file})

  set(run_target buddy-nv-buddynext-decode-${mlir_base}-run)
  add_executable(${run_target} ${MODEL_BUDDYNEXT_DIR}/buddy-next-runtime.cpp)
  add_dependencies(${run_target} ${lib_target})
  target_link_directories(${run_target} PRIVATE ${NV_LLVM_LIBRARY_DIR})
  target_link_libraries(${run_target}
    ${lib_target}
    NvHostCRunnerUtils
    ${NV_MLIR_C_RUNNER_UTILS_LIB}
    ${NV_MLIR_RUNNER_UTILS_LIB}
    ${NV_MLIR_ASYNC_RUNTIME_LIB}
    ${NV_MLIR_CUDA_RUNTIME_LIB}
    dl
    pthread
  )
  list(APPEND BUDDYNEXT_DECODE_TARGETS ${run_target})
endforeach()

add_custom_target(buddy-nv-buddynext-prefill-run DEPENDS ${BUDDYNEXT_PREFILL_TARGETS})
add_custom_target(buddy-nv-buddynext-decode-run DEPENDS ${BUDDYNEXT_DECODE_TARGETS})
add_custom_target(buddy-nv-buddynext-all-run
  DEPENDS buddy-nv-buddynext-prefill-run buddy-nv-buddynext-decode-run)
