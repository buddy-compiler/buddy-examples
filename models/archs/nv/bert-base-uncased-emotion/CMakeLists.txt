set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

add_custom_command(
  OUTPUT ${OUTPUT_DIR}/forward.mlir
         ${OUTPUT_DIR}/subgraph0.mlir
         ${OUTPUT_DIR}/arg0.data
         ${OUTPUT_DIR}/bert-parameters.h
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
  COMMAND ${PYTHON} buddy-pytorch-importer.py
            --repo-root ${REPO_ROOT}
            --output-dir ${OUTPUT_DIR}
            --compiler-build ${COMPILER_BUILD}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing bert-base-uncased-emotion for nv"
  VERBATIM
)

# Host x86 + NVIDIA GPU path (copied from nv/LeNet, adapted).

# ${OUTPUT_DIR}/forward.mlir is already memref glue (like BuddyBert); keep it on host CPU.
add_custom_command(
  OUTPUT forward_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_gpu_llvm.mlir
  COMMAND ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt ${OUTPUT_DIR}/forward.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg)" |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -pass-pipeline "builtin.module(func.func(buffer-deallocation-simplification, convert-linalg-to-loops), eliminate-empty-tensors, func.func(llvm-request-c-wrappers), convert-math-to-llvm, convert-math-to-libm, convert-scf-to-cf, convert-cf-to-llvm, convert-arith-to-llvm, expand-strided-metadata, finalize-memref-to-llvm, convert-func-to-llvm, reconcile-unrealized-casts)"
            -o ${OUTPUT_DIR}/forward_gpu_llvm.mlir
  COMMAND ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt ${OUTPUT_DIR}/forward.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg)" |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -pass-pipeline "builtin.module(func.func(buffer-deallocation-simplification, convert-linalg-to-loops), eliminate-empty-tensors, func.func(llvm-request-c-wrappers), convert-math-to-llvm, convert-math-to-libm, convert-scf-to-cf, convert-cf-to-llvm, convert-arith-to-llvm, expand-strided-metadata, finalize-memref-to-llvm, convert-func-to-llvm, reconcile-unrealized-casts)" |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-translate -mlir-to-llvmir |
          ${BUDDY_SOURCE}/llvm/build/bin/llvm-as |
          ${BUDDY_SOURCE}/llvm/build/bin/llc -filetype=obj -relocation-model=pic -O2 -o ${CMAKE_CURRENT_BINARY_DIR}/forward_gpu.o
  DEPENDS ${OUTPUT_DIR}/forward.mlir
  COMMENT "Building NV forward_gpu.o"
  VERBATIM
)

# Bert GPU path is broken: process-args=0 passes host strided memrefs into
# gpu.launch_func (CUDA_ERROR_ILLEGAL_ADDRESS); process-args=1 breaks subview
# layouts. Lower subgraph on host CPU (sequential, no OpenMP) until that is fixed.
add_custom_command(
  OUTPUT subgraph0_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -convert-trace-to-llvm |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -convert-elementwise-to-linalg
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -matmul-parallel-vectorization-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -lower-affine
            -convert-vector-to-scf
            -expand-strided-metadata
            -lower-affine
            -cse
            -convert-vector-to-llvm
            -memref-expand
            -arith-expand
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/subgraph0_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -convert-trace-to-llvm |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-opt
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -convert-elementwise-to-linalg
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -matmul-parallel-vectorization-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -lower-affine
            -convert-vector-to-scf
            -expand-strided-metadata
            -lower-affine
            -cse
            -convert-vector-to-llvm
            -memref-expand
            -arith-expand
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${BUDDY_SOURCE}/llvm/build/bin/mlir-translate -mlir-to-llvmir |
          ${BUDDY_SOURCE}/llvm/build/bin/llvm-as |
          ${BUDDY_SOURCE}/llvm/build/bin/llc -filetype=obj -relocation-model=pic -O2 -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_gpu.o
  DEPENDS ${OUTPUT_DIR}/subgraph0.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
  COMMENT "Building NV subgraph0_gpu.o (host CPU, no OpenMP)"
  VERBATIM
)

add_library(bert-base-uncased-emotion-nv-model STATIC EXCLUDE_FROM_ALL subgraph0_gpu.o forward_gpu.o)
set_target_properties(bert-base-uncased-emotion-nv-model PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
set_target_properties(bert-base-uncased-emotion-nv-model PROPERTIES LINKER_LANGUAGE CXX)

add_executable(buddy-nv-bert-run EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
set_target_properties(buddy-nv-bert-run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
add_dependencies(buddy-nv-bert-run bert-base-uncased-emotion-nv-model)
target_include_directories(buddy-nv-bert-run PRIVATE ${BUDDY_SOURCE}/frontend/Interfaces
                                                     ${MODEL_DIR}/bert-base-uncased-emotion
)
target_link_directories(buddy-nv-bert-run PRIVATE ${BUDDY_SOURCE}/llvm/build/lib ${BUDDY_SOURCE}/build/lib)
target_link_libraries(buddy-nv-bert-run bert-base-uncased-emotion-nv-model nv-runtime
                                                                           PNG::PNG
                                                                           BuddyLibDIP
                                                                           mlir_c_runner_utils
                                                                           mlir_runner_utils
                                                                           mlir_async_runtime
                                                                           mlir_cuda_runtime
                                                                           dl
                                                                           pthread
)

add_custom_target(bert-base-uncased-emotion-model-build DEPENDS bert-base-uncased-emotion-nv-model)

target_include_directories(buddy-nv-bert-run PRIVATE "${OUTPUT_DIR}")
