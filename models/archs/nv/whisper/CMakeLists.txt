set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

add_custom_command(
  OUTPUT ${OUTPUT_DIR}/forward.mlir
         ${OUTPUT_DIR}/subgraph0.mlir
         ${OUTPUT_DIR}/arg0.data
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
  COMMAND ${PYTHON} buddy-pytorch-importer.py
            --repo-root ${REPO_ROOT}
            --output-dir ${OUTPUT_DIR}
            --compiler-build ${COMPILER_BUILD}
            --spec ${MODEL_DIR}/whisper/specs/base.json
            --no-layer-partitioned
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing whisper for nv"
  VERBATIM
)

# Host x86 + NVIDIA GPU path (copied from nv/LeNet, adapted).

add_custom_command(
  OUTPUT forward_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${CMAKE_BINARY_DIR}/models/Whisper/forward.mlir
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
            -o ${OUTPUT_DIR}/forward_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${CMAKE_BINARY_DIR}/models/Whisper/forward.mlir
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
          ${BUDDY_SOURCE}/llvm/build/bin/llc -filetype=obj -relocation-model=pic -O2 -o ${CMAKE_CURRENT_BINARY_DIR}/forward_gpu.o
  DEPENDS ${CMAKE_BINARY_DIR}/models/Whisper/forward.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
  COMMENT "Building NV forward_gpu.o"
  VERBATIM
)

add_custom_command(
  OUTPUT subgraph0_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_gpu_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_gpu_gpu.mlir
             ${OUTPUT_DIR}/subgraph0_gpu_nvvm.mlir
             ${OUTPUT_DIR}/subgraph0_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${CMAKE_BINARY_DIR}/models/Whisper/subgraph0.mlir
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
            -o ${OUTPUT_DIR}/subgraph0_gpu_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${CMAKE_BINARY_DIR}/models/Whisper/subgraph0.mlir
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
            -o ${OUTPUT_DIR}/subgraph0_gpu_nvvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${CMAKE_BINARY_DIR}/models/Whisper/subgraph0.mlir
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
            -o ${OUTPUT_DIR}/subgraph0_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${CMAKE_BINARY_DIR}/models/Whisper/subgraph0.mlir
            -convert-trace-to-llvm |
          ${COMPILER_BUILD}/bin/buddy-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_gpu_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${CMAKE_BINARY_DIR}/models/Whisper/subgraph0.mlir
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
          ${BUDDY_SOURCE}/llvm/build/bin/llc -filetype=obj -relocation-model=pic -O2 -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_gpu.o
  DEPENDS ${CMAKE_BINARY_DIR}/models/Whisper/subgraph0.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
  COMMENT "Building NV subgraph0_gpu.o (TOSA→NVVM)"
  VERBATIM
)

add_library(whisper-nv-model STATIC EXCLUDE_FROM_ALL subgraph0_gpu.o forward_gpu.o)
set_target_properties(whisper-nv-model PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
set_target_properties(whisper-nv-model PROPERTIES LINKER_LANGUAGE CXX)

add_executable(buddy-nv-whisper-run EXCLUDE_FROM_ALL ${MODEL_DIR}/whisper/main.cpp)
set_target_properties(buddy-nv-whisper-run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
add_dependencies(buddy-nv-whisper-run whisper-nv-model)
target_include_directories(buddy-nv-whisper-run PRIVATE ${BUDDY_SOURCE}/frontend/Interfaces
                                                        ${MODEL_DIR}/whisper
)
target_link_directories(buddy-nv-whisper-run PRIVATE ${BUDDY_SOURCE}/llvm/build/lib ${BUDDY_SOURCE}/build/lib)
target_link_libraries(buddy-nv-whisper-run whisper-nv-model nv-runtime
                                                            BuddyLibDAP
                                                            mlir_c_runner_utils
                                                            mlir_runner_utils
                                                            mlir_async_runtime
                                                            mlir_cuda_runtime
                                                            dl
                                                            pthread
)

add_custom_target(whisper-model-build DEPENDS whisper-nv-model)
