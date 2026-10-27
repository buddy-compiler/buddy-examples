set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

# ===----------------------------------------------------------------------===
# Import the MLP-Mixer graph and partition it. No goban Ball is recognized yet, so
# the graph stays FP32: no layout alignment and no quantization.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/01-import.txt
         ${OUTPUT_DIR}/04-partitioned.txt
         ${OUTPUT_DIR}/subgraph0.mlir
         ${OUTPUT_DIR}/forward.mlir
         ${OUTPUT_DIR}/arg0.data
         ${OUTPUT_DIR}/input.f32
  COMMAND ${PYTHON} buddy-pytorch-importer.py
  DEPENDS buddy-pytorch-importer.py
          trace/trace.toml
          ${MODEL_DIR}/mlp-mixer/model.py
          ${MODEL_DIR}/mlp-mixer/output/mlp-mixer/model.safetensors
          ${MODEL_DIR}/alexnet/images/dog.bmp
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing MLP-Mixer"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower the model entry function to a RISC-V object.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward.o
  BYPRODUCTS ${OUTPUT_DIR}/forward.ll
             ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUTPUT_DIR}/forward.mlir
            -buffer-deallocation-simplification
            -convert-linalg-to-loops
            -eliminate-empty-tensors
            -llvm-request-c-wrappers
            -convert-math-to-llvm
            -convert-math-to-libm
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -expand-strided-metadata
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/forward_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/forward.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/forward.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o forward.o
  DEPENDS ${OUTPUT_DIR}/forward.mlir
          ${DESIGN_TARGET_PLUGIN_matmul}
          ${DESIGN_COMPILER}/bin/buddy-translate
          ${DESIGN_COMPILER}/bin/buddy-llc
  COMMENT "Building forward.o"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower the compute graph and export intermediate MLIR. The matmul thread has
# no recognition pass yet: everything runs in software on the goban core.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0.ll
             ${OUTPUT_DIR}/subgraph0_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_buckyball.mlir
             ${OUTPUT_DIR}/subgraph0_physical.mlir
             ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUTPUT_DIR}/subgraph0.mlir
            -allow-unregistered-dialect
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_linalg.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUTPUT_DIR}/subgraph0_linalg.mlir
            -allow-unregistered-dialect
            -lower-frontend
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -o ${OUTPUT_DIR}/subgraph0_buckyball.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUTPUT_DIR}/subgraph0_buckyball.mlir
            -convert-trace-to-llvm
            -assign-physical-banks
            -llvm-request-c-wrappers
            -o ${OUTPUT_DIR}/subgraph0_physical.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUTPUT_DIR}/subgraph0_physical.mlir
            -lower-buckyball
            -expand-strided-metadata
            -convert-linalg-to-loops
            -canonicalize
            -cse
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/subgraph0.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o subgraph0.o
  DEPENDS ${OUTPUT_DIR}/subgraph0.mlir
          ${DESIGN_TARGET_PLUGIN_matmul}
          ${DESIGN_COMPILER}/bin/buddy-translate
          ${DESIGN_COMPILER}/bin/buddy-llc
  COMMENT "Building subgraph0.o"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Link the model and runtime into a RISC-V shared library.
# ===----------------------------------------------------------------------===
add_library(mlp-mixer-buckyball-goban-model SHARED EXCLUDE_FROM_ALL forward.o subgraph0.o)
set_target_properties(mlp-mixer-buckyball-goban-model
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             LINKER_LANGUAGE CXX
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_link_libraries(mlp-mixer-buckyball-goban-model
  PRIVATE runtime-linux-shared
)

# ===----------------------------------------------------------------------===
# Build the Buddy InferenceRunner plugin for RISC-V Linux.
# ===----------------------------------------------------------------------===
get_filename_component(FLATBUFFERS_HEADER "${TOOLCHAIN}/include/flatbuffers/flatbuffers.h" REALPATH)
get_filename_component(FLATBUFFERS_INCLUDE "${FLATBUFFERS_HEADER}/../.." ABSOLUTE)
add_library(mlp-mixer-buckyball-goban-runner SHARED EXCLUDE_FROM_ALL runner.cpp)
set_target_properties(mlp-mixer-buckyball-goban-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_include_directories(mlp-mixer-buckyball-goban-runner
  PRIVATE ${LIB_DIR}
          ${FLATBUFFERS_INCLUDE}
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/runtime/include
          ${DESIGN_COMPILER}/runtime/include
)
target_link_libraries(mlp-mixer-buckyball-goban-runner
  PRIVATE dl
)

# ===----------------------------------------------------------------------===
# Package the RAX
#
# Contents:
# RISC-V model library (libmlp-mixer-buckyball-goban-model.so),
# RISC-V runner library (libmlp-mixer-buckyball-goban-runner.so),
# FP32 parameters (arg0.data).
#
# Output: output/mlp-mixer.rax embeds these files alongside RHAL metadata
# identifying the runner, model entry and parameter buffers.
# ===----------------------------------------------------------------------===
configure_file(mlp-mixer.rhal.mlir.in ${OUTPUT_DIR}/mlp-mixer.rhal.mlir @ONLY)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/mlp-mixer.rax
  COMMAND ${DESIGN_COMPILER}/bin/rax-pack ${OUTPUT_DIR}/mlp-mixer.rhal.mlir
          --embed-payload
          -o ${OUTPUT_DIR}/mlp-mixer.rax
  DEPENDS ${OUTPUT_DIR}/mlp-mixer.rhal.mlir
          ${DESIGN_COMPILER}/bin/rax-pack
          mlp-mixer-buckyball-goban-model
          mlp-mixer-buckyball-goban-runner
          ${OUTPUT_DIR}/arg0.data
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging mlp-mixer for buckyball-goban"
  VERBATIM
)
add_custom_target(mlp-mixer-model-build DEPENDS ${OUTPUT_DIR}/mlp-mixer.rax)
