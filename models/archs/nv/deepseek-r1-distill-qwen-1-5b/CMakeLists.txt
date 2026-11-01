file(GLOB_RECURSE RECIPE_FRONTEND_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/frontend/*.py")
file(GLOB MODEL_WEIGHTS CONFIGURE_DEPENDS
  "${MODEL_DIR}/deepseek-r1-distill-qwen-1-5b/output/deepseek-r1-distill-qwen-1-5b/*.safetensors"
  "${MODEL_DIR}/deepseek-r1-distill-qwen-1-5b/output/deepseek-r1-distill-qwen-1-5b/*.bin"
)

set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

add_custom_command(
  OUTPUT ${OUTPUT_DIR}/01-import-prefill.txt
         ${OUTPUT_DIR}/02-permuted-prefill.txt
         ${OUTPUT_DIR}/forward_prefill.mlir
         ${OUTPUT_DIR}/subgraph0_prefill.mlir
         ${OUTPUT_DIR}/01-import-decode.txt
         ${OUTPUT_DIR}/02-permuted-decode.txt
         ${OUTPUT_DIR}/forward_decode.mlir
         ${OUTPUT_DIR}/subgraph0_decode.mlir
         ${OUTPUT_DIR}/arg0.data
         ${OUTPUT_DIR}/arg1.data
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
  COMMAND ${PYTHON} buddy-pytorch-importer.py
  DEPENDS ${RECIPE_FRONTEND_SOURCES}
          ${COMPILER_FRONTEND_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${MODEL_DIR}/deepseek-r1-distill-qwen-1-5b/model.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing deepseek-r1-distill-qwen-1-5b for nv"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Compile the host entry point. Each command must succeed before the next runs.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward_prefill_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_prefill_linalg.mlir
             ${OUTPUT_DIR}/forward_prefill_gpu.mlir
             ${OUTPUT_DIR}/forward_prefill_gpu_llvm.mlir
             ${OUTPUT_DIR}/forward_prefill_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_prefill.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/forward_prefill_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_prefill_linalg.mlir
            -pass-pipeline "builtin.module(func.func(buffer-hoisting),canonicalize,cse,expand-strided-metadata,func.func(convert-memcpy-to-gpu,gpu-async-region))"
            -o ${OUTPUT_DIR}/forward_prefill_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_prefill_gpu.mlir
            -llvm-request-c-wrappers
            -gpu-to-llvm
            -o ${OUTPUT_DIR}/forward_prefill_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/forward_prefill_gpu_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/forward_prefill_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward_prefill_gpu.ll -filetype=obj -relocation-model=pic -O2 -o forward_prefill_gpu.o
  DEPENDS ${OUTPUT_DIR}/forward_prefill.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling deepseek-r1-distill-qwen-1-5b host entry point of forward_prefill"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower model computation to GPU kernels and embed the NVIDIA binary.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0_prefill_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_prefill_gpu_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_prefill_gpu_weights.mlir
             ${OUTPUT_DIR}/subgraph0_prefill_gpu_frontend.mlir
             ${OUTPUT_DIR}/subgraph0_prefill_bufferized.mlir
             ${OUTPUT_DIR}/subgraph0_prefill_copies.mlir
             ${OUTPUT_DIR}/subgraph0_prefill_gpu_gpu.mlir
             ${OUTPUT_DIR}/subgraph0_prefill_gpu_async.mlir
             ${OUTPUT_DIR}/subgraph0_prefill_gpu_llvm.mlir
             ${OUTPUT_DIR}/subgraph0_prefill_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_prefill.mlir
            --load-dialect-plugin=${GENERIC_COMPILER}
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_prefill_gpu_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_prefill_gpu_linalg.mlir
            --load-dialect-plugin=${NV_COMPILER}
            --load-pass-plugin=${NV_COMPILER}
            -fuse-mxint8-weights
            -o ${OUTPUT_DIR}/subgraph0_prefill_gpu_weights.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_prefill_gpu_weights.mlir
            --load-dialect-plugin=${GENERIC_COMPILER}
            --load-pass-plugin=${GENERIC_COMPILER}
            -lower-frontend-fake-op=gpu=true
            -o ${OUTPUT_DIR}/subgraph0_prefill_gpu_frontend.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_prefill_gpu_frontend.mlir
            -convert-elementwise-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -o ${OUTPUT_DIR}/subgraph0_prefill_bufferized.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_prefill_bufferized.mlir
            -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))"
            -o ${OUTPUT_DIR}/subgraph0_prefill_copies.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_prefill_copies.mlir
            -convert-linalg-to-parallel-loops "-scf-parallel-loop-tiling=parallel-loop-tile-sizes=8,4,32 no-min-max-bounds=true"
            -gpu-map-parallel-loops=mapping-policy=innermost-first
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse
            -o ${OUTPUT_DIR}/subgraph0_prefill_gpu_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_prefill_gpu_gpu.mlir
            -pass-pipeline "builtin.module(func.func(convert-memcpy-to-gpu{process-args=0},gpu-async-region),canonicalize)"
            -o ${OUTPUT_DIR}/subgraph0_prefill_gpu_async.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_prefill_gpu_async.mlir
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=${NV_CUBIN_VERSION} cubin-features=+ptx71 cubin-format=fatbin"
            -o ${OUTPUT_DIR}/subgraph0_prefill_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_prefill_gpu_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/subgraph0_prefill_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0_prefill_gpu.ll -filetype=obj -relocation-model=pic -O2 -o subgraph0_prefill_gpu.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_prefill.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${GENERIC_COMPILER}
          ${NV_COMPILER}
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling deepseek-r1-distill-qwen-1-5b NVIDIA kernels of forward_prefill"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Compile the host entry point. Each command must succeed before the next runs.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward_decode_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_decode_linalg.mlir
             ${OUTPUT_DIR}/forward_decode_gpu.mlir
             ${OUTPUT_DIR}/forward_decode_gpu_llvm.mlir
             ${OUTPUT_DIR}/forward_decode_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_decode.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/forward_decode_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_decode_linalg.mlir
            -pass-pipeline "builtin.module(func.func(buffer-hoisting),canonicalize,cse,expand-strided-metadata,func.func(convert-memcpy-to-gpu,gpu-async-region))"
            -o ${OUTPUT_DIR}/forward_decode_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_decode_gpu.mlir
            -llvm-request-c-wrappers
            -gpu-to-llvm
            -o ${OUTPUT_DIR}/forward_decode_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/forward_decode_gpu_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/forward_decode_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward_decode_gpu.ll -filetype=obj -relocation-model=pic -O2 -o forward_decode_gpu.o
  DEPENDS ${OUTPUT_DIR}/forward_decode.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling deepseek-r1-distill-qwen-1-5b host entry point of forward_decode"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower model computation to GPU kernels and embed the NVIDIA binary.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0_decode_gpu.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_decode_gpu_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_decode_gpu_weights.mlir
             ${OUTPUT_DIR}/subgraph0_decode_gpu_frontend.mlir
             ${OUTPUT_DIR}/subgraph0_decode_bufferized.mlir
             ${OUTPUT_DIR}/subgraph0_decode_copies.mlir
             ${OUTPUT_DIR}/subgraph0_decode_gpu_gpu.mlir
             ${OUTPUT_DIR}/subgraph0_decode_gpu_async.mlir
             ${OUTPUT_DIR}/subgraph0_decode_gpu_llvm.mlir
             ${OUTPUT_DIR}/subgraph0_decode_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_decode.mlir
            --load-dialect-plugin=${GENERIC_COMPILER}
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_decode_gpu_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_decode_gpu_linalg.mlir
            --load-dialect-plugin=${NV_COMPILER}
            --load-pass-plugin=${NV_COMPILER}
            -fuse-mxint8-weights
            -o ${OUTPUT_DIR}/subgraph0_decode_gpu_weights.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_decode_gpu_weights.mlir
            --load-dialect-plugin=${GENERIC_COMPILER}
            --load-pass-plugin=${GENERIC_COMPILER}
            -lower-frontend-fake-op=gpu=true
            -o ${OUTPUT_DIR}/subgraph0_decode_gpu_frontend.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_decode_gpu_frontend.mlir
            -convert-elementwise-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -o ${OUTPUT_DIR}/subgraph0_decode_bufferized.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_decode_bufferized.mlir
            -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))"
            -o ${OUTPUT_DIR}/subgraph0_decode_copies.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_decode_copies.mlir
            -convert-linalg-to-parallel-loops "-scf-parallel-loop-tiling=parallel-loop-tile-sizes=8,4,32 no-min-max-bounds=true"
            -gpu-map-parallel-loops=mapping-policy=innermost-first
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse
            -o ${OUTPUT_DIR}/subgraph0_decode_gpu_gpu.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_decode_gpu_gpu.mlir
            -pass-pipeline "builtin.module(func.func(convert-memcpy-to-gpu{process-args=0},gpu-async-region),canonicalize)"
            -o ${OUTPUT_DIR}/subgraph0_decode_gpu_async.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_decode_gpu_async.mlir
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=${NV_CUBIN_VERSION} cubin-features=+ptx71 cubin-format=fatbin"
            -o ${OUTPUT_DIR}/subgraph0_decode_gpu_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_decode_gpu_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/subgraph0_decode_gpu.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0_decode_gpu.ll -filetype=obj -relocation-model=pic -O2 -o subgraph0_decode_gpu.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_decode.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${GENERIC_COMPILER}
          ${NV_COMPILER}
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling deepseek-r1-distill-qwen-1-5b NVIDIA kernels of forward_decode"
  VERBATIM
)

add_library(deepseek-r1-distill-qwen-1-5b-nv-custom-model SHARED EXCLUDE_FROM_ALL forward_prefill_gpu.o
  subgraph0_prefill_gpu.o forward_decode_gpu.o subgraph0_decode_gpu.o
)
set_target_properties(deepseek-r1-distill-qwen-1-5b-nv-custom-model
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             LINKER_LANGUAGE CXX
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_link_directories(deepseek-r1-distill-qwen-1-5b-nv-custom-model
  PRIVATE ${BUDDY_SOURCE}/llvm/build/lib
)
target_link_libraries(deepseek-r1-distill-qwen-1-5b-nv-custom-model
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
add_library(deepseek-r1-distill-qwen-1-5b-nv-custom-runner SHARED EXCLUDE_FROM_ALL runner.cpp)
set_target_properties(deepseek-r1-distill-qwen-1-5b-nv-custom-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_include_directories(deepseek-r1-distill-qwen-1-5b-nv-custom-runner
  PRIVATE ${LIB_DIR}
          ${OUTPUT_DIR}
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/runtime/include
          ${COMPILER_BUILD}/runtime/include
)
target_link_directories(deepseek-r1-distill-qwen-1-5b-nv-custom-runner
  PRIVATE ${BUDDY_SOURCE}/llvm/build/lib
)
target_link_libraries(deepseek-r1-distill-qwen-1-5b-nv-custom-runner
  PRIVATE dl
)

# ===----------------------------------------------------------------------===
# Package the RAX
#
# Contents:
# model library (libdeepseek-r1-distill-qwen-1-5b-nv-custom-model.so),
# runner library (libdeepseek-r1-distill-qwen-1-5b-nv-custom-runner.so),
# FP32 parameters (arg0.data), INT8 MXINT8 weights (arg1.data),
# token vocabulary (vocab.txt),
# MLIR runtime libraries (libmlir_c_runner_utils.so, libmlir_runner_utils.so,
# libmlir_async_runtime.so, libmlir_cuda_runtime.so).
#
# Output: output/deepseek-r1-distill-qwen-1-5b.rax embeds these files alongside RHAL metadata that
# identifies the runner, model library, parameters and runtime dependencies.
#
# ===----------------------------------------------------------------------===
configure_file(deepseek-r1-distill-qwen-1-5b.rhal.mlir.in ${OUTPUT_DIR}/deepseek-r1-distill-qwen-1-5b.rhal.mlir @ONLY)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/deepseek-r1-distill-qwen-1-5b.rax
  COMMAND ${COMPILER_BUILD}/bin/rax-pack ${OUTPUT_DIR}/deepseek-r1-distill-qwen-1-5b.rhal.mlir
          -o ${OUTPUT_DIR}/deepseek-r1-distill-qwen-1-5b.rax
          --embed-payload
  DEPENDS ${OUTPUT_DIR}/deepseek-r1-distill-qwen-1-5b.rhal.mlir
          deepseek-r1-distill-qwen-1-5b-nv-custom-model
          deepseek-r1-distill-qwen-1-5b-nv-custom-runner
          ${COMPILER_BUILD}/bin/rax-pack
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
          ${MODEL_DIR}/deepseek-r1-distill-qwen-1-5b/vocab.txt
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_c_runner_utils.so
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_runner_utils.so
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_async_runtime.so
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_cuda_runtime.so
  COMMENT "Packaging NVIDIA deepseek-r1-distill-qwen-1-5b as RAX"
  VERBATIM
)
add_custom_target(
  buddy-nv-deepseek-r1-distill-qwen-1-5b-rax-custom-build
  DEPENDS ${OUTPUT_DIR}/deepseek-r1-distill-qwen-1-5b.rax
)

add_subdirectory(origin)
