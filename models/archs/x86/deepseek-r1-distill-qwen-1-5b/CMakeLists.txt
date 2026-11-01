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
          ${MODEL_DIR}/deepseek-r1-distill-qwen-1-5b/output/deepseek-r1-distill-qwen-1-5b/config.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing deepseek-r1-distill-qwen-1-5b for x86"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Compile the host entry point to native x86 code.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward_prefill.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_prefill_linalg.mlir
             ${OUTPUT_DIR}/forward_prefill_llvm.mlir
             ${OUTPUT_DIR}/forward_prefill.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_prefill.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/forward_prefill_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_prefill_linalg.mlir
            -convert-elementwise-to-linalg
            -convert-linalg-to-loops
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/forward_prefill_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/forward_prefill_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/forward_prefill.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward_prefill.ll -filetype=obj -relocation-model=pic -O2 -o forward_prefill.o
  DEPENDS ${OUTPUT_DIR}/forward_prefill.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling deepseek-r1-distill-qwen-1-5b forward_prefill for x86"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower model computation to native x86 code. The parallel loops of every op
# run on an OpenMP team; reductions stay in order, so the results do not
# depend on the thread count. The prefill takes any number of
# tokens: its loops have dynamic bounds, which LLVM analyzes across the whole
# function, so every loop nest becomes a function of its own.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0_prefill.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_prefill_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_prefill_weights.mlir
             ${OUTPUT_DIR}/subgraph0_prefill_frontend.mlir
             ${OUTPUT_DIR}/subgraph0_prefill_loops.mlir
             ${OUTPUT_DIR}/subgraph0_prefill_outlined.mlir
             ${OUTPUT_DIR}/subgraph0_prefill_llvm.mlir
             ${OUTPUT_DIR}/subgraph0_prefill.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_prefill.mlir
            --load-dialect-plugin=${GENERIC_COMPILER}
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_prefill_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_prefill_linalg.mlir
            --load-dialect-plugin=${X86_COMPILER}
            --load-pass-plugin=${X86_COMPILER}
            -fuse-mxint8-weights
            -o ${OUTPUT_DIR}/subgraph0_prefill_weights.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_prefill_weights.mlir
            --load-dialect-plugin=${GENERIC_COMPILER}
            --load-pass-plugin=${GENERIC_COMPILER}
            -lower-frontend-fake-op
            -o ${OUTPUT_DIR}/subgraph0_prefill_frontend.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_prefill_frontend.mlir
            -convert-elementwise-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-bufferization-to-memref
            -convert-linalg-to-parallel-loops
            -o ${OUTPUT_DIR}/subgraph0_prefill_loops.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_prefill_loops.mlir
            -outline-loop-nests
            -o ${OUTPUT_DIR}/subgraph0_prefill_outlined.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_prefill_outlined.mlir
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-openmp
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-vector-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/subgraph0_prefill_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_prefill_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/subgraph0_prefill.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0_prefill.ll -filetype=obj -relocation-model=pic -O2 -o subgraph0_prefill.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_prefill.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${GENERIC_COMPILER}
          ${X86_COMPILER}
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling deepseek-r1-distill-qwen-1-5b subgraph0_prefill for x86"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Compile the host entry point to native x86 code.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward_decode.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_decode_linalg.mlir
             ${OUTPUT_DIR}/forward_decode_llvm.mlir
             ${OUTPUT_DIR}/forward_decode.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_decode.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/forward_decode_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_decode_linalg.mlir
            -convert-elementwise-to-linalg
            -convert-linalg-to-loops
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/forward_decode_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/forward_decode_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/forward_decode.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward_decode.ll -filetype=obj -relocation-model=pic -O2 -o forward_decode.o
  DEPENDS ${OUTPUT_DIR}/forward_decode.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling deepseek-r1-distill-qwen-1-5b forward_decode for x86"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower model computation to native x86 code. The parallel loops of every op
# run on an OpenMP team; reductions stay in order, so the results do not
# depend on the thread count.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0_decode.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_decode_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_decode_weights.mlir
             ${OUTPUT_DIR}/subgraph0_decode_frontend.mlir
             ${OUTPUT_DIR}/subgraph0_decode_llvm.mlir
             ${OUTPUT_DIR}/subgraph0_decode.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_decode.mlir
            --load-dialect-plugin=${GENERIC_COMPILER}
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_decode_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_decode_linalg.mlir
            --load-dialect-plugin=${X86_COMPILER}
            --load-pass-plugin=${X86_COMPILER}
            -fuse-mxint8-weights
            -o ${OUTPUT_DIR}/subgraph0_decode_weights.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_decode_weights.mlir
            --load-dialect-plugin=${GENERIC_COMPILER}
            --load-pass-plugin=${GENERIC_COMPILER}
            -lower-frontend-fake-op
            -o ${OUTPUT_DIR}/subgraph0_decode_frontend.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_decode_frontend.mlir
            -convert-elementwise-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-bufferization-to-memref
            -convert-linalg-to-parallel-loops
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-openmp
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-vector-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/subgraph0_decode_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_decode_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/subgraph0_decode.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0_decode.ll -filetype=obj -relocation-model=pic -O2 -o subgraph0_decode.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_decode.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${GENERIC_COMPILER}
          ${X86_COMPILER}
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling deepseek-r1-distill-qwen-1-5b subgraph0_decode for x86"
  VERBATIM
)

add_library(deepseek-r1-distill-qwen-1-5b-x86-custom-model SHARED EXCLUDE_FROM_ALL forward_prefill.o
  subgraph0_prefill.o forward_decode.o subgraph0_decode.o
)
set_target_properties(deepseek-r1-distill-qwen-1-5b-x86-custom-model
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             LINKER_LANGUAGE CXX
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_link_directories(deepseek-r1-distill-qwen-1-5b-x86-custom-model
  PRIVATE ${BUDDY_SOURCE}/llvm/build/lib
          ${BUDDY_SOURCE}/llvm/build/runtimes/runtimes-bins/openmp/runtime/src
)
target_link_libraries(deepseek-r1-distill-qwen-1-5b-x86-custom-model
  PRIVATE x86-runtime
          omp
          mlir_c_runner_utils
          mlir_runner_utils
          dl
          pthread
)

# ===----------------------------------------------------------------------===
# Build the Buddy InferenceRunner plugin.
# ===----------------------------------------------------------------------===
add_library(deepseek-r1-distill-qwen-1-5b-x86-custom-runner SHARED EXCLUDE_FROM_ALL runner.cpp)
set_target_properties(deepseek-r1-distill-qwen-1-5b-x86-custom-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_include_directories(deepseek-r1-distill-qwen-1-5b-x86-custom-runner
  PRIVATE ${LIB_DIR}
          ${OUTPUT_DIR}
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/runtime/include
          ${COMPILER_BUILD}/runtime/include
)
target_link_directories(deepseek-r1-distill-qwen-1-5b-x86-custom-runner
  PRIVATE ${BUDDY_SOURCE}/llvm/build/lib
)
target_link_libraries(deepseek-r1-distill-qwen-1-5b-x86-custom-runner
  PRIVATE dl
)

# ===----------------------------------------------------------------------===
# Package the RAX
#
# Contents:
# model library (libdeepseek-r1-distill-qwen-1-5b-x86-custom-model.so),
# runner library (libdeepseek-r1-distill-qwen-1-5b-x86-custom-runner.so),
# FP32 parameters (arg0.data), INT8 MXINT8 weights (arg1.data),
# token vocabulary (vocab.txt),
# MLIR runtime libraries (libmlir_c_runner_utils.so, libmlir_runner_utils.so).
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
          deepseek-r1-distill-qwen-1-5b-x86-custom-model
          deepseek-r1-distill-qwen-1-5b-x86-custom-runner
          ${COMPILER_BUILD}/bin/rax-pack
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
          ${MODEL_DIR}/deepseek-r1-distill-qwen-1-5b/vocab.txt
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_c_runner_utils.so
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_runner_utils.so
          ${BUDDY_SOURCE}/llvm/build/runtimes/runtimes-bins/openmp/runtime/src/libomp.so
  COMMENT "Packaging x86 deepseek-r1-distill-qwen-1-5b as RAX"
  VERBATIM
)
add_custom_target(
  buddy-x86-deepseek-r1-distill-qwen-1-5b-rax-custom-build
  DEPENDS ${OUTPUT_DIR}/deepseek-r1-distill-qwen-1-5b.rax
)

add_subdirectory(origin)
