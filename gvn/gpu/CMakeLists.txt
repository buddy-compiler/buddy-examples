# ===----------------------------------------------------------------------===
# The GPU side of a GVN run: the model's graph with the checked values
# exposed, compiled for the local NVIDIA GPU into a program that writes them.
# The same for every model; `bbdev gvn --debug` builds it on a GPU machine.
#
# Input: GVN_GPU, the run's gpu/ directory, where the driver puts
#   linalg.mlir   the model's linalg graph
#   ids           the locations of the values to expose, comma separated
#   forward.mlir  the model entry
#   gpu.cpp       the model's host harness
# Output there: gpu, the program, and layout.json, where it writes each value.
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.22)
project(gvn-gpu NONE)
get_filename_component(STACK_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(BUDDY "${STACK_ROOT}/thirdparty/buddy-mlir")
set(PLUGINS "${STACK_ROOT}/gvn/build")

# The compute capability of GPU 0: 7.5 -> sm_75.
execute_process(
  COMMAND nvidia-smi --id=0 --query-gpu=compute_cap --format=csv,noheader
  OUTPUT_VARIABLE CHIP
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
string(REPLACE "." "" CHIP "sm_${CHIP}")
file(READ ${GVN_GPU}/ids IDS)

# The graph returning the values at the ids.
add_custom_command(
  OUTPUT ${GVN_GPU}/exposed.mlir
         ${GVN_GPU}/layout.json
  COMMAND ${BUDDY}/build/bin/buddy-opt
            --load-dialect-plugin=${PLUGINS}/generic-compiler/libgeneric-compiler.so
            --load-pass-plugin=${PLUGINS}/generic-compiler/libgeneric-compiler.so
            --load-dialect-plugin=${PLUGINS}/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${PLUGINS}/gvn-compiler/libgvn-compiler.so ${GVN_GPU}/linalg.mlir "-gvn-expose=ids=${IDS} layout=${GVN_GPU}/layout.json"
            -o ${GVN_GPU}/exposed.mlir
  DEPENDS ${GVN_GPU}/linalg.mlir
          ${GVN_GPU}/ids
  VERBATIM
)

add_custom_command(
  OUTPUT ${GVN_GPU}/subgraph.o
  BYPRODUCTS ${GVN_GPU}/frontend.mlir
             ${GVN_GPU}/bufferized.mlir
             ${GVN_GPU}/copies.mlir
             ${GVN_GPU}/operators.mlir
             ${GVN_GPU}/gpu.mlir
             ${GVN_GPU}/async.mlir
             ${GVN_GPU}/subgraph_llvm.mlir
             ${GVN_GPU}/subgraph.ll
  COMMAND ${BUDDY}/build/bin/buddy-opt
            --load-dialect-plugin=${PLUGINS}/generic-compiler/libgeneric-compiler.so
            --load-pass-plugin=${PLUGINS}/generic-compiler/libgeneric-compiler.so
            --load-dialect-plugin=${PLUGINS}/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${PLUGINS}/gvn-compiler/libgvn-compiler.so ${GVN_GPU}/exposed.mlir
            -lower-frontend-fake-op=gpu=true
            -o ${GVN_GPU}/frontend.mlir
  COMMAND ${BUDDY}/build/bin/buddy-opt
            --load-dialect-plugin=${PLUGINS}/generic-compiler/libgeneric-compiler.so
            --load-pass-plugin=${PLUGINS}/generic-compiler/libgeneric-compiler.so
            --load-dialect-plugin=${PLUGINS}/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${PLUGINS}/gvn-compiler/libgvn-compiler.so ${GVN_GPU}/frontend.mlir
            -convert-elementwise-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-bufferization-to-memref
            -o ${GVN_GPU}/bufferized.mlir
  COMMAND ${BUDDY}/build/bin/buddy-opt
            --load-dialect-plugin=${PLUGINS}/generic-compiler/libgeneric-compiler.so
            --load-pass-plugin=${PLUGINS}/generic-compiler/libgeneric-compiler.so
            --load-dialect-plugin=${PLUGINS}/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${PLUGINS}/gvn-compiler/libgvn-compiler.so ${GVN_GPU}/bufferized.mlir
            -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))"
            -o ${GVN_GPU}/copies.mlir
  COMMAND ${BUDDY}/build/bin/buddy-opt
            --load-dialect-plugin=${PLUGINS}/nv-compiler/libnv-compiler.so
            --load-pass-plugin=${PLUGINS}/nv-compiler/libnv-compiler.so ${GVN_GPU}/copies.mlir
            -linalg-generalize-named-ops
            -linalg-to-gpu-kernels
            -o ${GVN_GPU}/operators.mlir
  COMMAND ${BUDDY}/build/bin/buddy-opt
            --load-dialect-plugin=${PLUGINS}/generic-compiler/libgeneric-compiler.so
            --load-pass-plugin=${PLUGINS}/generic-compiler/libgeneric-compiler.so
            --load-dialect-plugin=${PLUGINS}/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${PLUGINS}/gvn-compiler/libgvn-compiler.so ${GVN_GPU}/operators.mlir
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-launch-sink-index-computations
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse
            -o ${GVN_GPU}/gpu.mlir
  COMMAND ${BUDDY}/build/bin/buddy-opt
            --load-dialect-plugin=${PLUGINS}/generic-compiler/libgeneric-compiler.so
            --load-pass-plugin=${PLUGINS}/generic-compiler/libgeneric-compiler.so
            --load-dialect-plugin=${PLUGINS}/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${PLUGINS}/gvn-compiler/libgvn-compiler.so ${GVN_GPU}/gpu.mlir
            -pass-pipeline "builtin.module(func.func(convert-memcpy-to-gpu{process-args=0},gpu-async-region,gvn-managed-memory),canonicalize)"
            -o ${GVN_GPU}/async.mlir
  COMMAND ${BUDDY}/build/bin/buddy-opt
            --load-dialect-plugin=${PLUGINS}/generic-compiler/libgeneric-compiler.so
            --load-pass-plugin=${PLUGINS}/generic-compiler/libgeneric-compiler.so
            --load-dialect-plugin=${PLUGINS}/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${PLUGINS}/gvn-compiler/libgvn-compiler.so ${GVN_GPU}/async.mlir
            -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=${CHIP} cubin-features=+ptx71 cubin-format=fatbin"
            -o ${GVN_GPU}/subgraph_llvm.mlir
  COMMAND ${BUDDY}/build/bin/buddy-translate ${GVN_GPU}/subgraph_llvm.mlir -mlir-to-llvmir -o ${GVN_GPU}/subgraph.ll
  COMMAND ${BUDDY}/build/bin/buddy-llc ${GVN_GPU}/subgraph.ll -filetype=obj -relocation-model=pic -O2 -o ${GVN_GPU}/subgraph.o
  DEPENDS ${GVN_GPU}/exposed.mlir
  VERBATIM
)

add_custom_command(
  OUTPUT ${GVN_GPU}/forward.o
  BYPRODUCTS ${GVN_GPU}/forward_exposed.mlir
             ${GVN_GPU}/forward_gpu.mlir
             ${GVN_GPU}/forward_llvm.mlir
             ${GVN_GPU}/forward.ll
  COMMAND ${BUDDY}/build/bin/buddy-opt
            --load-dialect-plugin=${PLUGINS}/generic-compiler/libgeneric-compiler.so
            --load-pass-plugin=${PLUGINS}/generic-compiler/libgeneric-compiler.so
            --load-dialect-plugin=${PLUGINS}/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${PLUGINS}/gvn-compiler/libgvn-compiler.so ${GVN_GPU}/forward.mlir
            -gvn-forward-results=layout=${GVN_GPU}/layout.json
            -o ${GVN_GPU}/forward_exposed.mlir
  COMMAND ${BUDDY}/build/bin/buddy-opt
            --load-dialect-plugin=${PLUGINS}/generic-compiler/libgeneric-compiler.so
            --load-pass-plugin=${PLUGINS}/generic-compiler/libgeneric-compiler.so
            --load-dialect-plugin=${PLUGINS}/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${PLUGINS}/gvn-compiler/libgvn-compiler.so ${GVN_GPU}/forward_exposed.mlir
            -pass-pipeline "builtin.module(func.func(buffer-hoisting),canonicalize,cse,expand-strided-metadata,func.func(convert-memcpy-to-gpu,gpu-async-region,gvn-managed-memory))"
            -o ${GVN_GPU}/forward_gpu.mlir
  COMMAND ${BUDDY}/build/bin/buddy-opt
            --load-dialect-plugin=${PLUGINS}/generic-compiler/libgeneric-compiler.so
            --load-pass-plugin=${PLUGINS}/generic-compiler/libgeneric-compiler.so
            --load-dialect-plugin=${PLUGINS}/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${PLUGINS}/gvn-compiler/libgvn-compiler.so ${GVN_GPU}/forward_gpu.mlir
            -llvm-request-c-wrappers
            -gpu-to-llvm
            -o ${GVN_GPU}/forward_llvm.mlir
  COMMAND ${BUDDY}/build/bin/buddy-translate ${GVN_GPU}/forward_llvm.mlir -mlir-to-llvmir -o ${GVN_GPU}/forward.ll
  COMMAND ${BUDDY}/build/bin/buddy-llc ${GVN_GPU}/forward.ll -filetype=obj -relocation-model=pic -O2 -o ${GVN_GPU}/forward.o
  DEPENDS ${GVN_GPU}/forward.mlir
          ${GVN_GPU}/layout.json
  VERBATIM
)

add_custom_command(
  OUTPUT ${GVN_GPU}/gpu
  COMMAND ${STACK_ROOT}/result/bin/g++ -std=c++17 -O2 -I${STACK_ROOT}/models/lib -I${STACK_ROOT}/runtime/trace -I${BUDDY}/llvm/mlir/include/mlir/ExecutionEngine ${GVN_GPU}/gpu.cpp ${GVN_GPU}/forward.o ${GVN_GPU}/subgraph.o ${STACK_ROOT}/models/lib/CRunnerUtils.cpp ${STACK_ROOT}/models/lib/native_trace.cpp ${STACK_ROOT}/runtime/trace/trace.cpp ${STACK_ROOT}/runtime/trace/cycle_file.cpp -L${BUDDY}/llvm/build/lib -Wl,-rpath,${BUDDY}/llvm/build/lib -lmlir_c_runner_utils -lmlir_runner_utils -lmlir_async_runtime -lmlir_cuda_runtime -ldl -lpthread -o ${GVN_GPU}/gpu
  DEPENDS ${GVN_GPU}/gpu.cpp
          ${GVN_GPU}/forward.o
          ${GVN_GPU}/subgraph.o
  VERBATIM
)
add_custom_target(gpu ALL DEPENDS ${GVN_GPU}/gpu)
