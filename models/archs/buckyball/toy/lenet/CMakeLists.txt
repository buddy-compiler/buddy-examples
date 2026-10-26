file(GLOB_RECURSE RECIPE_FRONTEND_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/frontend/*.py")

set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

# ===----------------------------------------------------------------------===
# Import LeNet into MLIR and export its quantized model data.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/forward.mlir
         ${OUTPUT_DIR}/subgraph0.mlir
         ${OUTPUT_DIR}/arg0.data
         ${OUTPUT_DIR}/arg1.data
  COMMAND ${PYTHON} buddy-pytorch-importer.py
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          ${RECIPE_FRONTEND_SOURCES}
          buddy-pytorch-importer.py
          ${MODEL_DIR}/lenet/output/lenet/model.pth
          quant/quant.toml
          trace/trace.toml
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower the model entry function to a RISC-V object.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward.o
  BYPRODUCTS ${OUTPUT_DIR}/forward.ll
             ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${DESIGN_TARGET_OPT_toy} ${OUTPUT_DIR}/forward.mlir -buffer-deallocation-simplification -convert-linalg-to-loops -eliminate-empty-tensors -llvm-request-c-wrappers -convert-math-to-llvm -convert-math-to-libm -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -expand-strided-metadata -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts -o ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/forward_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/forward.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/forward.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o forward.o
  DEPENDS "${OUTPUT_DIR}/forward.mlir"
          ${DESIGN_TARGET_PLUGIN_toy}
          ${DESIGN_COMPILER}/bin/buddy-translate
          ${DESIGN_COMPILER}/bin/buddy-llc
  COMMENT "Building forward.o"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower the compute graphs to RISC-V objects and export intermediate MLIR.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0.ll
             ${OUTPUT_DIR}/subgraph0_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_bank_ssa.mlir
             ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${DESIGN_TARGET_OPT_toy} ${OUTPUT_DIR}/subgraph0.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" -o ${OUTPUT_DIR}/subgraph0_linalg.mlir
  COMMAND ${DESIGN_TARGET_OPT_toy} ${OUTPUT_DIR}/subgraph0_linalg.mlir -eliminate-empty-tensors -convert-elementwise-to-linalg -one-shot-bufferize=bufferize-function-boundaries -convert-linalg-to-gemmini-ball -lower-buckyball-to-bank-ssa -assign-physical-banks -o ${OUTPUT_DIR}/subgraph0_bank_ssa.mlir
  COMMAND ${DESIGN_TARGET_OPT_toy} ${OUTPUT_DIR}/subgraph0_bank_ssa.mlir -llvm-request-c-wrappers -convert-trace-to-llvm -lower-buckyball -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -buffer-deallocation-simplification -bufferization-lower-deallocations -convert-math-to-llvm -convert-math-to-libm -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts -o ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/subgraph0.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o subgraph0.o
  DEPENDS "${OUTPUT_DIR}/subgraph0.mlir"
          ${DESIGN_TARGET_PLUGIN_toy}
          ${DESIGN_COMPILER}/bin/buddy-translate
          ${DESIGN_COMPILER}/bin/buddy-llc
  COMMENT "Building subgraph0.o"
  VERBATIM
)
# ===----------------------------------------------------------------------===
# Link the model and runtime into a RISC-V shared library.
# ===----------------------------------------------------------------------===
add_library(lenet-buckyball-toy-model SHARED EXCLUDE_FROM_ALL forward.o subgraph0.o)
set_target_properties(lenet-buckyball-toy-model
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             LINKER_LANGUAGE CXX
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_link_libraries(lenet-buckyball-toy-model
  PRIVATE runtime-linux-shared
)

# ===----------------------------------------------------------------------===
# Build the Buddy InferenceRunner plugin for RISC-V Linux.
# ===----------------------------------------------------------------------===
get_filename_component(FLATBUFFERS_HEADER "${TOOLCHAIN}/include/flatbuffers/flatbuffers.h" REALPATH)
get_filename_component(FLATBUFFERS_INCLUDE "${FLATBUFFERS_HEADER}/../.." ABSOLUTE)
add_library(lenet-buckyball-toy-runner SHARED EXCLUDE_FROM_ALL runner.cpp)
set_target_properties(lenet-buckyball-toy-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_include_directories(lenet-buckyball-toy-runner
  PRIVATE ${LIB_DIR}
          ${FLATBUFFERS_INCLUDE}
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/runtime/include
          ${DESIGN_COMPILER}/runtime/include
)
target_link_libraries(lenet-buckyball-toy-runner
  PRIVATE dl
)

# ===----------------------------------------------------------------------===
# Package the RAX
#
# Contents:
# RISC-V model library (liblenet-buckyball-toy-model.so),
# RISC-V runner library (liblenet-buckyball-toy-runner.so),
# FP32 parameters (arg0.data), and
# INT8 weights (arg1.data).
#
# Output: output/lenet.rax embeds these files alongside RHAL metadata
# identifying the runner, model entry and parameter buffers.
# ===----------------------------------------------------------------------===
configure_file(lenet.rhal.mlir.in ${OUTPUT_DIR}/lenet.rhal.mlir @ONLY)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/lenet.rax
  COMMAND ${DESIGN_COMPILER}/bin/rax-pack ${OUTPUT_DIR}/lenet.rhal.mlir
          --embed-payload
          -o ${OUTPUT_DIR}/lenet.rax
  DEPENDS ${OUTPUT_DIR}/lenet.rhal.mlir
          ${DESIGN_COMPILER}/bin/rax-pack
          lenet-buckyball-toy-model
          lenet-buckyball-toy-runner
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging lenet for buckyball-toy"
  VERBATIM
)
add_custom_target(lenet-model-build DEPENDS ${OUTPUT_DIR}/lenet.rax)
