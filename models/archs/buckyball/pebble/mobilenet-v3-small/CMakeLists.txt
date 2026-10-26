file(GLOB_RECURSE RECIPE_FRONTEND_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/frontend/*.py")

set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")
set(PAYLOAD_DIR "${OUTPUT_DIR}/mobilenet-v3-small.payload")

# ===----------------------------------------------------------------------===
# Calibrate the activation scales on ImageNet training images.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/calibration.json
  COMMAND ${PYTHON} buddy-pytorch-calibrate.py
  DEPENDS buddy-pytorch-calibrate.py
          frontend/quant/graph.py
          ${COMPILER_FRONTEND_SOURCES}
          ${MODEL_DIR}/mobilenet-v3-small/output/mobilenet-v3-small/model.pth
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Calibrating MobileNetV3-Small activations"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Import the MobileNetV3-Small graph, permute, quantize and partition it.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/01-import.txt
         ${OUTPUT_DIR}/02-permuted.txt
         ${OUTPUT_DIR}/03-quanted.txt
         ${OUTPUT_DIR}/04-partitioned.txt
         ${OUTPUT_DIR}/subgraph0.mlir
         ${OUTPUT_DIR}/forward.mlir
         ${OUTPUT_DIR}/arg0.data
         ${OUTPUT_DIR}/arg1.data
         ${OUTPUT_DIR}/arg2.data
  COMMAND ${PYTHON} buddy-pytorch-importer.py
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          ${RECIPE_FRONTEND_SOURCES}
          buddy-pytorch-importer.py
          trace/trace.toml
          ${OUTPUT_DIR}/calibration.json
          ${MODEL_DIR}/mobilenet-v3-small/output/mobilenet-v3-small/model.pth
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing MobileNetV3-Small"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower the model entry function to a RISC-V object.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward.o
  BYPRODUCTS ${OUTPUT_DIR}/forward.ll
             ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/forward.mlir -buffer-deallocation-simplification -convert-linalg-to-loops -eliminate-empty-tensors -llvm-request-c-wrappers -convert-math-to-llvm -convert-math-to-libm -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-math-to-llvm -convert-math-to-libm -convert-arith-to-llvm -expand-strided-metadata -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts -o ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/forward_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/forward.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/forward.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o forward.o
  DEPENDS ${OUTPUT_DIR}/forward.mlir
          ${DESIGN_TARGET_PLUGIN_pebble}
          ${DESIGN_COMPILER}/bin/buddy-translate
          ${DESIGN_COMPILER}/bin/buddy-llc
  COMMENT "Building forward.o"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower the compute graph and export intermediate MLIR.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0.ll
             ${OUTPUT_DIR}/subgraph0_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_custom.mlir
             ${OUTPUT_DIR}/subgraph0_buckyball.mlir
             ${OUTPUT_DIR}/subgraph0_physical.mlir
             ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/subgraph0.mlir -allow-unregistered-dialect -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" -o ${OUTPUT_DIR}/subgraph0_linalg.mlir
  COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/subgraph0_linalg.mlir -allow-unregistered-dialect -form-custom-ops -o ${OUTPUT_DIR}/subgraph0_custom.mlir
  COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/subgraph0_custom.mlir -allow-unregistered-dialect -convert-linalg-to-buckyball -lower-frontend -eliminate-empty-tensors -convert-elementwise-to-linalg -one-shot-bufferize=bufferize-function-boundaries -ownership-based-buffer-deallocation -buffer-deallocation-simplification -bufferization-lower-deallocations -o ${OUTPUT_DIR}/subgraph0_buckyball.mlir
  COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/subgraph0_buckyball.mlir -convert-trace-to-llvm -assign-physical-banks -llvm-request-c-wrappers -o ${OUTPUT_DIR}/subgraph0_physical.mlir
  COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/subgraph0_physical.mlir -lower-buckyball -expand-strided-metadata -convert-linalg-to-loops -canonicalize -cse -lower-affine -convert-scf-to-cf=allow-pattern-rollback=false -convert-cf-to-llvm -convert-math-to-llvm -convert-math-to-libm -convert-arith-to-llvm -convert-func-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/subgraph0.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o subgraph0.o
  DEPENDS ${OUTPUT_DIR}/subgraph0.mlir
          ${DESIGN_TARGET_PLUGIN_pebble}
          ${DESIGN_COMPILER}/bin/buddy-translate
          ${DESIGN_COMPILER}/bin/buddy-llc
  COMMENT "Building subgraph0.o"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Link the model and runtime into a RISC-V shared library.
# ===----------------------------------------------------------------------===
add_library(mobilenet-v3-small-buckyball-pebble-model SHARED EXCLUDE_FROM_ALL forward.o subgraph0.o)
set_target_properties(mobilenet-v3-small-buckyball-pebble-model
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             LINKER_LANGUAGE CXX
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_link_libraries(mobilenet-v3-small-buckyball-pebble-model
  PRIVATE runtime-linux-shared
)

# ===----------------------------------------------------------------------===
# Build the Buddy InferenceRunner plugin for RISC-V Linux.
# ===----------------------------------------------------------------------===
get_filename_component(FLATBUFFERS_HEADER "${TOOLCHAIN}/include/flatbuffers/flatbuffers.h" REALPATH)
get_filename_component(FLATBUFFERS_INCLUDE "${FLATBUFFERS_HEADER}/../.." ABSOLUTE)
add_library(mobilenet-v3-small-buckyball-pebble-runner SHARED EXCLUDE_FROM_ALL runner.cpp)
set_target_properties(mobilenet-v3-small-buckyball-pebble-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_include_directories(mobilenet-v3-small-buckyball-pebble-runner
  PRIVATE ${LIB_DIR}
          ${FLATBUFFERS_INCLUDE}
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/runtime/include
          ${DESIGN_COMPILER}/runtime/include
)
target_link_libraries(mobilenet-v3-small-buckyball-pebble-runner
  PRIVATE dl
          model_image
)

# ===----------------------------------------------------------------------===
# Package the RAX
#
# Contents:
# RISC-V model library (libmobilenet-v3-small-buckyball-pebble-model.so),
# RISC-V runner library (libmobilenet-v3-small-buckyball-pebble-runner.so),
# FP32 parameters and scales (arg0.data), INT32 biases (arg1.data),
# INT8 weights (arg2.data), and the ImageNet class names (Labels.txt).
#
# Output: output/mobilenet-v3-small.rax embeds these files alongside RHAL metadata
# identifying the runner, model entry and parameter buffers.
# ===----------------------------------------------------------------------===
configure_file(mobilenet-v3-small.rhal.mlir.in ${OUTPUT_DIR}/mobilenet-v3-small.rhal.mlir @ONLY)
configure_file(${MODEL_DIR}/mobilenet-v3-small/Labels.txt ${OUTPUT_DIR}/Labels.txt COPYONLY)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/mobilenet-v3-small.rax
  COMMAND ${DESIGN_COMPILER}/bin/rax-pack ${OUTPUT_DIR}/mobilenet-v3-small.rhal.mlir
          --embed-payload
          -o ${OUTPUT_DIR}/mobilenet-v3-small.rax
  DEPENDS ${OUTPUT_DIR}/mobilenet-v3-small.rhal.mlir
          ${DESIGN_COMPILER}/bin/rax-pack
          mobilenet-v3-small-buckyball-pebble-model
          mobilenet-v3-small-buckyball-pebble-runner
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
          ${OUTPUT_DIR}/arg2.data
          ${OUTPUT_DIR}/Labels.txt
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging mobilenet-v3-small for buckyball-pebble"
  VERBATIM
)
add_custom_target(mobilenet-v3-small-model-build DEPENDS ${OUTPUT_DIR}/mobilenet-v3-small.rax)
