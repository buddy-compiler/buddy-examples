set(CMAKE_CXX_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

set(MODEL_STABLEDIFFUSION_DIR ${MODEL_DIR}/StableDiffusion)
set(INTERFACE_DIR ${BUDDY_MLIR_DIR}/frontend/Interfaces)
set(INCLUDE_DIR ${MODEL_STABLEDIFFUSION_DIR}/include)
set(TOSA_PIPELINE "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))")

function(add_sd_object output mlir trace)
  set(TRACE_PASS)
  if(trace)
    set(TRACE_PASS -convert-trace-to-llvm="cycle-trace")
  endif()

  add_custom_command(
    OUTPUT ${output}
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/${mlir}
              -pass-pipeline ${TOSA_PIPELINE} |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -eliminate-empty-tensors
              -eliminate-large-zero-constants
              -empty-tensor-to-alloc-tensor
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -one-shot-bufferize="bufferize-function-boundaries"
              -llvm-request-c-wrappers
              ${TRACE_PASS}
              -expand-strided-metadata
              -convert-linalg-to-loops
              -buffer-deallocation-simplification
              -bufferization-lower-deallocations
              -convert-vector-to-scf
              -lower-affine
              -convert-scf-to-cf
              -convert-cf-to-llvm
              -convert-cf-to-llvm
              -convert-vector-to-llvm
              -convert-math-to-llvm
              -convert-math-to-libm
              -convert-arith-to-llvm
              -convert-func-to-llvm
              -finalize-memref-to-llvm
              -reconcile-unrealized-casts |
            ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
            ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard -o ${output}
    DEPENDS ${MODEL_STABLEDIFFUSION_DIR}/${mlir}
    COMMENT "Building ${output}"
    VERBATIM
    COMMAND_EXPAND_LISTS)
endfunction()

add_sd_object(forward_text_encoder.o forward_text_encoder.mlir FALSE)
add_sd_object(subgraph0_text_encoder.o subgraph0_text_encoder.mlir TRUE)
add_sd_object(forward_unet.o forward_unet.mlir FALSE)
add_sd_object(subgraph0_unet.o subgraph0_unet.mlir TRUE)
add_sd_object(forward_vae.o forward_vae.mlir FALSE)
add_sd_object(subgraph0_vae.o subgraph0_vae.mlir TRUE)

add_library(BuckyballTEXTENCODER STATIC subgraph0_text_encoder.o forward_text_encoder.o)
add_library(BuckyballUNET STATIC subgraph0_unet.o forward_unet.o)
add_library(BuckyballVAE STATIC subgraph0_vae.o forward_vae.o)

set_target_properties(BuckyballTEXTENCODER PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(BuckyballUNET PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(BuckyballVAE PROPERTIES LINKER_LANGUAGE CXX)

add_dependencies(BuckyballTEXTENCODER stable-diffusion-model-build)
add_dependencies(BuckyballUNET stable-diffusion-model-build)
add_dependencies(BuckyballVAE stable-diffusion-model-build)

add_executable(buddy-buckyball-stable-diffusion-run ${MODEL_STABLEDIFFUSION_DIR}/buddy-stable-diffusion-main.cpp)

target_include_directories(buddy-buckyball-stable-diffusion-run PRIVATE ${INTERFACE_DIR} ${INCLUDE_DIR})

if(BUDDY_MLIR_USE_MIMALLOC)
  list(APPEND BUDDY_STABLE_DIFFUSION_LIBS mimalloc)
endif()

set(BUDDY_STABLE_DIFFUSION_LIBS
  BuckyballTEXTENCODER
  BuckyballUNET
  BuckyballVAE
  ModelTestCRunnerUtils
  ModelTestDIP_riscv
)

target_link_libraries(buddy-buckyball-stable-diffusion-run -static ${BUDDY_STABLE_DIFFUSION_LIBS})

set(STABLEDIFFUSION_OUTPUT_DIR ${OUTPUT_BIN_DIR}/src/ModelTest/e2e/models/archs/buckyball/StableDiffusion)

add_custom_command(TARGET buddy-buckyball-stable-diffusion-run POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${STABLEDIFFUSION_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:buddy-buckyball-stable-diffusion-run>
          ${STABLEDIFFUSION_OUTPUT_DIR}/buddy-buckyball-stable-diffusion-run
  COMMENT "Copying Buckyball StableDiffusion executable to output directory"
  VERBATIM
)

add_custom_target(sync-buckyball-stable-diffusion-deps
  COMMAND ${CMAKE_COMMAND} -E make_directory ${STABLEDIFFUSION_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${STABLEDIFFUSION_OUTPUT_DIR}/trace/cycle
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_STABLEDIFFUSION_DIR}/arg0_text_encoder.data
          ${STABLEDIFFUSION_OUTPUT_DIR}/arg0_text_encoder.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_STABLEDIFFUSION_DIR}/arg1_text_encoder.data
          ${STABLEDIFFUSION_OUTPUT_DIR}/arg1_text_encoder.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_STABLEDIFFUSION_DIR}/arg0_unet.data
          ${STABLEDIFFUSION_OUTPUT_DIR}/arg0_unet.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_STABLEDIFFUSION_DIR}/arg0_vae.data
          ${STABLEDIFFUSION_OUTPUT_DIR}/arg0_vae.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_STABLEDIFFUSION_DIR}/vocab.txt
          ${STABLEDIFFUSION_OUTPUT_DIR}/vocab.txt
  DEPENDS
          ${MODEL_STABLEDIFFUSION_DIR}/arg0_text_encoder.data
          ${MODEL_STABLEDIFFUSION_DIR}/arg1_text_encoder.data
          ${MODEL_STABLEDIFFUSION_DIR}/arg0_unet.data
          ${MODEL_STABLEDIFFUSION_DIR}/arg0_vae.data
          ${MODEL_STABLEDIFFUSION_DIR}/vocab.txt
  COMMENT "Syncing StableDiffusion runtime deps to output directory"
  VERBATIM
)

add_dependencies(buddy-buckyball-stable-diffusion-run sync-buckyball-stable-diffusion-deps)
