set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

add_custom_command(
  OUTPUT ${OUTPUT_DIR}/arg0_text_encoder.data
         ${OUTPUT_DIR}/arg1_text_encoder.data
         ${OUTPUT_DIR}/arg0_unet.data
         ${OUTPUT_DIR}/arg0_vae.data
         ${OUTPUT_DIR}/forward_text_encoder.mlir
         ${OUTPUT_DIR}/subgraph0_text_encoder.mlir
         ${OUTPUT_DIR}/forward_unet.mlir
         ${OUTPUT_DIR}/subgraph0_unet.mlir
         ${OUTPUT_DIR}/forward_vae.mlir
         ${OUTPUT_DIR}/subgraph0_vae.mlir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
  COMMAND ${PYTHON} buddy-pytorch-importer.py
            --checkpoint "${MODEL_DIR}/stable-diffusion/output/stable-diffusion"
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${MODEL_DIR}/stable-diffusion/output/stable-diffusion/model_index.json
          ${CMAKE_CURRENT_SOURCE_DIR}/trace/text_encoder.toml
          ${CMAKE_CURRENT_SOURCE_DIR}/trace/unet.toml
          ${CMAKE_CURRENT_SOURCE_DIR}/trace/vae.toml
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing stable-diffusion for x86"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Compile the host entry point to native x86 code.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward_text_encoder.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_text_encoder_trace.mlir
             ${OUTPUT_DIR}/forward_text_encoder_linalg.mlir
             ${OUTPUT_DIR}/forward_text_encoder_llvm.mlir
             ${OUTPUT_DIR}/forward_text_encoder.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_text_encoder.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/forward_text_encoder_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_text_encoder_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/forward_text_encoder_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_text_encoder_linalg.mlir
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
            -o ${OUTPUT_DIR}/forward_text_encoder_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/forward_text_encoder_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/forward_text_encoder.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward_text_encoder.ll -filetype=obj -relocation-model=pic -O3 -o forward_text_encoder.o
  DEPENDS ${OUTPUT_DIR}/forward_text_encoder.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling stable-diffusion forward_text_encoder for x86"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower model computation to native x86 code.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0_text_encoder.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_text_encoder_trace.mlir
             ${OUTPUT_DIR}/subgraph0_text_encoder_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_text_encoder_llvm.mlir
             ${OUTPUT_DIR}/subgraph0_text_encoder.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/subgraph0_text_encoder_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_text_encoder_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_text_encoder_linalg.mlir
            -convert-elementwise-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
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
            -o ${OUTPUT_DIR}/subgraph0_text_encoder_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_text_encoder_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/subgraph0_text_encoder.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0_text_encoder.ll -filetype=obj -relocation-model=pic -O3 -o subgraph0_text_encoder.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_text_encoder.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling stable-diffusion subgraph0_text_encoder for x86"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Compile the host entry point to native x86 code.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward_unet.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_unet_trace.mlir
             ${OUTPUT_DIR}/forward_unet_linalg.mlir
             ${OUTPUT_DIR}/forward_unet_llvm.mlir
             ${OUTPUT_DIR}/forward_unet.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_unet.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/forward_unet_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_unet_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/forward_unet_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_unet_linalg.mlir
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
            -o ${OUTPUT_DIR}/forward_unet_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/forward_unet_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/forward_unet.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward_unet.ll -filetype=obj -relocation-model=pic -O3 -o forward_unet.o
  DEPENDS ${OUTPUT_DIR}/forward_unet.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling stable-diffusion forward_unet for x86"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower model computation to native x86 code.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0_unet.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_unet_trace.mlir
             ${OUTPUT_DIR}/subgraph0_unet_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_unet_llvm.mlir
             ${OUTPUT_DIR}/subgraph0_unet.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_unet.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/subgraph0_unet_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_unet_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_unet_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_unet_linalg.mlir
            -convert-elementwise-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
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
            -o ${OUTPUT_DIR}/subgraph0_unet_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_unet_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/subgraph0_unet.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0_unet.ll -filetype=obj -relocation-model=pic -O3 -o subgraph0_unet.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_unet.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling stable-diffusion subgraph0_unet for x86"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Compile the host entry point to native x86 code.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward_vae.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_vae_trace.mlir
             ${OUTPUT_DIR}/forward_vae_linalg.mlir
             ${OUTPUT_DIR}/forward_vae_llvm.mlir
             ${OUTPUT_DIR}/forward_vae.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_vae.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/forward_vae_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_vae_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/forward_vae_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward_vae_linalg.mlir
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
            -o ${OUTPUT_DIR}/forward_vae_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/forward_vae_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/forward_vae.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/forward_vae.ll -filetype=obj -relocation-model=pic -O3 -o forward_vae.o
  DEPENDS ${OUTPUT_DIR}/forward_vae.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling stable-diffusion forward_vae for x86"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower model computation to native x86 code.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0_vae.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_vae_trace.mlir
             ${OUTPUT_DIR}/subgraph0_vae_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_vae_llvm.mlir
             ${OUTPUT_DIR}/subgraph0_vae.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_vae.mlir
            -convert-trace-to-llvm
            -o ${OUTPUT_DIR}/subgraph0_vae_trace.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_vae_trace.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_vae_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0_vae_linalg.mlir
            -convert-elementwise-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
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
            -o ${OUTPUT_DIR}/subgraph0_vae_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_vae_llvm.mlir -mlir-to-llvmir -o ${OUTPUT_DIR}/subgraph0_vae.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0_vae.ll -filetype=obj -relocation-model=pic -O3 -o subgraph0_vae.o
  DEPENDS ${OUTPUT_DIR}/subgraph0_vae.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Compiling stable-diffusion subgraph0_vae for x86"
  VERBATIM
)

add_library(stable-diffusion-x86-origin-model SHARED EXCLUDE_FROM_ALL forward_text_encoder.o subgraph0_text_encoder.o
  forward_unet.o subgraph0_unet.o forward_vae.o subgraph0_vae.o
)
set_target_properties(stable-diffusion-x86-origin-model
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             LINKER_LANGUAGE CXX
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_link_directories(stable-diffusion-x86-origin-model
  PRIVATE ${BUDDY_SOURCE}/llvm/build/lib
)
target_link_libraries(stable-diffusion-x86-origin-model
  PRIVATE x86-runtime
          mlir_c_runner_utils
          mlir_runner_utils
          dl
          pthread
)

# ===----------------------------------------------------------------------===
# Build the Buddy InferenceRunner plugin.
# ===----------------------------------------------------------------------===
add_library(stable-diffusion-x86-origin-runner SHARED EXCLUDE_FROM_ALL runner.cpp)
set_target_properties(stable-diffusion-x86-origin-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_include_directories(stable-diffusion-x86-origin-runner
  PRIVATE ${LIB_DIR}
          ${OUTPUT_DIR}
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/runtime/include
          ${COMPILER_BUILD}/runtime/include
)
target_link_directories(stable-diffusion-x86-origin-runner
  PRIVATE ${BUDDY_SOURCE}/llvm/build/lib
)
target_link_libraries(stable-diffusion-x86-origin-runner
  PRIVATE dl
          PNG::PNG
          ${COMPILER_BUILD}/lib/libBuddyLibDIP.a
          mlir_c_runner_utils
)

# ===----------------------------------------------------------------------===
# Package the RAX
#
# Contents:
# model library (libstable-diffusion-x86-origin-model.so),
# runner library (libstable-diffusion-x86-origin-runner.so),
# FP32 text encoder parameters (arg0_text_encoder.data),
# INT64 text positions (arg1_text_encoder.data),
# FP32 denoiser parameters (arg0_unet.data),
# FP32 VAE parameters (arg0_vae.data),
# token vocabulary (vocab.txt),
# MLIR runtime libraries (libmlir_c_runner_utils.so, libmlir_runner_utils.so).
#
# Output: output/stable-diffusion.rax embeds these files alongside RHAL metadata that
# identifies the runner, model library, parameters and runtime dependencies.
#
# ===----------------------------------------------------------------------===
configure_file(stable-diffusion.rhal.mlir.in ${OUTPUT_DIR}/stable-diffusion.rhal.mlir @ONLY)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/stable-diffusion.rax
  COMMAND ${COMPILER_BUILD}/bin/rax-pack ${OUTPUT_DIR}/stable-diffusion.rhal.mlir
          -o ${OUTPUT_DIR}/stable-diffusion.rax
          --embed-payload
  DEPENDS ${OUTPUT_DIR}/stable-diffusion.rhal.mlir
          stable-diffusion-x86-origin-model
          stable-diffusion-x86-origin-runner
          ${COMPILER_BUILD}/bin/rax-pack
          ${OUTPUT_DIR}/arg0_text_encoder.data
          ${OUTPUT_DIR}/arg1_text_encoder.data
          ${OUTPUT_DIR}/arg0_unet.data
          ${OUTPUT_DIR}/arg0_vae.data
          ${MODEL_DIR}/stable-diffusion/vocab.txt
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_c_runner_utils.so
          ${BUDDY_SOURCE}/llvm/build/lib/libmlir_runner_utils.so
  COMMENT "Packaging x86 stable-diffusion as RAX"
  VERBATIM
)
add_custom_target(buddy-x86-stable-diffusion-rax-origin-build DEPENDS ${OUTPUT_DIR}/stable-diffusion.rax)
