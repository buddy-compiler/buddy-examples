# ===----------------------------------------------------------------------===
# The GPU side of a GVN run of a device program: the programs and the linalg
# graphs they launch merged (gvn-merge), each launch a call recording its
# operands, its results and the values the slices name (gvn-gpu-program),
# compiled for the local NVIDIA GPU with managed memory, and run by the model
# recipe's host harness. The same for every design whose models are device
# programs; `bbdev gvn --debug` builds it on a GPU machine, again whenever
# what it reads changed.
#
# Input: DESIGN_CMAKE, the config's generated design variables; GVN_GPU, the
# run's gpu/ directory, where the driver puts values.json (the ids of each
# graph); MODULES, the programs and the graphs, in order; MODEL, the model's
# output directory; HARNESS, gvn/ of the model recipe (gpu.cpp, and what it
# reads besides; model.h above it).
# Output in GVN_GPU: program.json (the layout of the launches and values),
# program.txt (the manifest), values.bin and index.txt (the recorded values).
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.22)
project(gvn-program NONE)
include(${DESIGN_CMAKE})
get_filename_component(STACK_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(BUDDY "${STACK_ROOT}/thirdparty/buddy-mlir")
set(PLUGINS "${STACK_ROOT}/gvn/build")
set(
  GVN_PLUGIN
  --load-dialect-plugin=${PLUGINS}/gvn-compiler/libgvn-compiler.so
  --load-pass-plugin=${PLUGINS}/gvn-compiler/libgvn-compiler.so
)
set(
  GENERIC_OPT
  ${BUDDY}/build/bin/buddy-opt
  --load-dialect-plugin=${PLUGINS}/generic-compiler/libgeneric-compiler.so
  --load-pass-plugin=${PLUGINS}/generic-compiler/libgeneric-compiler.so
)

# The compute capability of GPU 0: 7.5 -> sm_75.
execute_process(
  COMMAND nvidia-smi --id=0 --query-gpu=compute_cap --format=csv,noheader
  OUTPUT_VARIABLE CHIP
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
string(REPLACE "." "" CHIP "sm_${CHIP}")
# The driver the CUDA runtime of the LLVM build loads.
execute_process(
  COMMAND ldd ${BUDDY}/llvm/build/lib/libmlir_cuda_runtime.so
  OUTPUT_VARIABLE LINKED
  COMMAND_ERROR_IS_FATAL ANY
)
string(REGEX MATCH "/nix/store/[^ ]*/libcuda\\.so[^ ]*" CUDA "${LINKED}")
get_filename_component(CUDA "${CUDA}" DIRECTORY)

# The programs and graphs printed with their locations, one module.
set(LOCAL)
foreach(module ${MODULES})
  get_filename_component(name ${module} NAME_WE)
  add_custom_command(
    OUTPUT ${GVN_GPU}/${name}.local.mlir
    COMMAND ${DESIGN_DEVICE_OPT} ${GVN_PLUGIN} ${module} -mlir-print-local-scope -mlir-print-debuginfo -o ${GVN_GPU}/${name}.local.mlir
    DEPENDS ${module}
            ${PLUGINS}/gvn-compiler/libgvn-compiler.so
    COMMAND_EXPAND_LISTS
    VERBATIM
  )
  list(APPEND LOCAL ${GVN_GPU}/${name}.local.mlir)
endforeach()

# Not VERBATIM: the shell redirects the concatenation.
add_custom_command(
  OUTPUT ${GVN_GPU}/modules.mlir
  COMMAND ${CMAKE_COMMAND} -E cat ${LOCAL} > ${GVN_GPU}/modules.mlir
  DEPENDS ${LOCAL}
)

add_custom_command(
  OUTPUT ${GVN_GPU}/recorded.mlir
         ${GVN_GPU}/program.json
         ${GVN_GPU}/program.txt
  COMMAND ${DESIGN_DEVICE_OPT} ${GVN_PLUGIN} ${GVN_GPU}/modules.mlir -verify-each=0 -gvn-merge "-gvn-gpu-program=values=${GVN_GPU}/values.json layout=${GVN_GPU}/program.json manifest=${GVN_GPU}/program.txt" -o ${GVN_GPU}/recorded.mlir
  DEPENDS ${GVN_GPU}/modules.mlir
          ${GVN_GPU}/values.json
  COMMAND_EXPAND_LISTS
  VERBATIM
)

add_custom_command(
  OUTPUT ${GVN_GPU}/program.o
  BYPRODUCTS ${GVN_GPU}/frontend.mlir
             ${GVN_GPU}/bufferized.mlir
             ${GVN_GPU}/copies_heap.mlir
             ${GVN_GPU}/copies.mlir
             ${GVN_GPU}/operators.mlir
             ${GVN_GPU}/gpu.mlir
             ${GVN_GPU}/managed.mlir
             ${GVN_GPU}/llvm.mlir
             ${GVN_GPU}/packed.mlir
             ${GVN_GPU}/program.ll
  COMMAND ${GENERIC_OPT} ${GVN_GPU}/recorded.mlir -lower-frontend-fake-op=gpu=true -o ${GVN_GPU}/frontend.mlir
  COMMAND ${GENERIC_OPT} ${GVN_GPU}/frontend.mlir -convert-elementwise-to-linalg -canonicalize -cse "-one-shot-bufferize=copy-before-write bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -ownership-based-buffer-deallocation -buffer-deallocation-simplification -bufferization-lower-deallocations -convert-bufferization-to-memref -o ${GVN_GPU}/bufferized.mlir
  COMMAND ${GENERIC_OPT} ${GVN_GPU}/bufferized.mlir -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${GVN_GPU}/copies_heap.mlir
  COMMAND ${BUDDY}/build/bin/buddy-opt ${GVN_PLUGIN} ${GVN_GPU}/copies_heap.mlir
            -pass-pipeline "builtin.module(gvn-host-shapes,func.func(convert-strided-memref-copy-to-linalg))"
            -o ${GVN_GPU}/copies.mlir
  COMMAND ${BUDDY}/build/bin/buddy-opt
            --load-dialect-plugin=${PLUGINS}/nv-compiler/libnv-compiler.so
            --load-pass-plugin=${PLUGINS}/nv-compiler/libnv-compiler.so ${GVN_GPU}/copies.mlir
            -linalg-generalize-named-ops
            -linalg-to-gpu-kernels
            -o ${GVN_GPU}/operators.mlir
  COMMAND ${GENERIC_OPT} ${GVN_GPU}/operators.mlir -canonicalize -gpu-map-parallel-loops -convert-parallel-loops-to-gpu -gpu-launch-sink-index-computations -gpu-kernel-outlining -buffer-hoisting -canonicalize -cse -o ${GVN_GPU}/gpu.mlir
  COMMAND ${BUDDY}/build/bin/buddy-opt ${GVN_PLUGIN} ${GVN_GPU}/gpu.mlir
            -pass-pipeline "builtin.module(func.func(gpu-async-region,gvn-managed-memory),canonicalize)"
            -o ${GVN_GPU}/managed.mlir
  # ptxas: the CUDA toolkit of the environment (the one compiled into
  # buddy-mlir is a garbage-collected nix path).
  COMMAND ${CMAKE_COMMAND} -E env CUDA_ROOT=${STACK_ROOT}/result ${GENERIC_OPT} ${GVN_GPU}/managed.mlir -llvm-request-c-wrappers "-gpu-lower-to-nvvm-pipeline=cubin-chip=${CHIP} cubin-features=+ptx71 cubin-format=fatbin" -o ${GVN_GPU}/llvm.mlir
  COMMAND ${BUDDY}/build/bin/buddy-opt ${GVN_PLUGIN} ${GVN_GPU}/llvm.mlir
            -gvn-packed-entry
            -o ${GVN_GPU}/packed.mlir
  COMMAND ${BUDDY}/build/bin/buddy-translate ${GVN_GPU}/packed.mlir -mlir-to-llvmir -o ${GVN_GPU}/program.ll
  # Host code launching the kernels (compiled with the NVVM pipeline):
  # optimizing it only costs time.
  COMMAND ${BUDDY}/build/bin/buddy-llc ${GVN_GPU}/program.ll -filetype=obj -relocation-model=pic -O0 -o ${GVN_GPU}/program.o
  DEPENDS ${GVN_GPU}/recorded.mlir
  COMMAND_EXPAND_LISTS
  VERBATIM
)

get_filename_component(RECIPE ${HARNESS} DIRECTORY)
add_custom_command(
  OUTPUT ${GVN_GPU}/gpu
  COMMAND ${STACK_ROOT}/result/bin/g++ -std=c++17 -O2 -I${RECIPE} -I${STACK_ROOT}/models/lib -I${STACK_ROOT}/runtime/trace -I${BUDDY}/llvm/mlir/include/mlir/ExecutionEngine ${HARNESS}/gpu.cpp ${GVN_GPU}/program.o ${STACK_ROOT}/models/lib/CRunnerUtils.cpp ${STACK_ROOT}/models/lib/native_trace.cpp ${STACK_ROOT}/runtime/trace/trace.cpp ${STACK_ROOT}/runtime/trace/cycle_file.cpp -L${BUDDY}/llvm/build/lib -Wl,-rpath,${BUDDY}/llvm/build/lib -L${CUDA} -Wl,-rpath,${CUDA} -lmlir_c_runner_utils -lmlir_runner_utils -lmlir_async_runtime -lmlir_cuda_runtime -lcuda -ldl -lpthread -lm -o ${GVN_GPU}/gpu
  DEPENDS ${HARNESS}/gpu.cpp
          ${GVN_GPU}/program.o
  VERBATIM
)

file(GLOB HARNESS_INPUTS ${HARNESS}/*)
add_custom_command(
  OUTPUT ${GVN_GPU}/values.bin
         ${GVN_GPU}/index.txt
  COMMAND ${GVN_GPU}/gpu ${GVN_GPU}/program.txt ${GVN_GPU}/values.bin ${GVN_GPU}/index.txt ${MODEL} ${HARNESS}
  DEPENDS ${GVN_GPU}/gpu
          ${GVN_GPU}/program.txt
          ${HARNESS_INPUTS}
  WORKING_DIRECTORY ${GVN_GPU}
  VERBATIM
)
add_custom_target(capture ALL DEPENDS ${GVN_GPU}/values.bin ${GVN_GPU}/index.txt)
