set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

# ===----------------------------------------------------------------------===
# Import the YOLO26n graph and partition it.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/01-import.txt
         ${OUTPUT_DIR}/02-partitioned.txt
         ${OUTPUT_DIR}/subgraph0.mlir
         ${OUTPUT_DIR}/forward.mlir
         ${OUTPUT_DIR}/arg0.data
  COMMAND ${PYTHON} buddy-pytorch-importer.py
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          buddy-pytorch-importer.py
          trace/trace.toml
          ${MODEL_DIR}/yolo26n/output/yolo26n/yolo26n.pt
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing YOLO26n"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower the model entry function to a RISC-V object.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT forward.o
  BYPRODUCTS ${OUTPUT_DIR}/forward.ll
             ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/forward.mlir
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
          ${DESIGN_TARGET_PLUGIN_pebble}
          ${DESIGN_COMPILER}/bin/buddy-translate
          ${DESIGN_COMPILER}/bin/buddy-llc
  COMMENT "Building forward.o"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower the compute graph in software and export intermediate MLIR.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0.ll
             ${OUTPUT_DIR}/subgraph0_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_bufferized.mlir
             ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/subgraph0.mlir
            -allow-unregistered-dialect
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_linalg.mlir
  COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/subgraph0_linalg.mlir
            -allow-unregistered-dialect
            -lower-frontend
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -o ${OUTPUT_DIR}/subgraph0_bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/subgraph0_bufferized.mlir
            -convert-trace-to-llvm
            -llvm-request-c-wrappers
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
          ${DESIGN_TARGET_PLUGIN_pebble}
          ${DESIGN_COMPILER}/bin/buddy-translate
          ${DESIGN_COMPILER}/bin/buddy-llc
  COMMENT "Building subgraph0.o"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Link the model and runtime into a RISC-V shared library.
# ===----------------------------------------------------------------------===
add_library(yolo26n-buckyball-pebble-model SHARED EXCLUDE_FROM_ALL forward.o subgraph0.o)
set_target_properties(yolo26n-buckyball-pebble-model
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             LINKER_LANGUAGE CXX
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_link_libraries(yolo26n-buckyball-pebble-model
  PRIVATE runtime-linux-shared
)

# ===----------------------------------------------------------------------===
# Build the Buddy InferenceRunner plugin for RISC-V Linux.
# ===----------------------------------------------------------------------===
get_filename_component(FLATBUFFERS_HEADER "${TOOLCHAIN}/include/flatbuffers/flatbuffers.h" REALPATH)
get_filename_component(FLATBUFFERS_INCLUDE "${FLATBUFFERS_HEADER}/../.." ABSOLUTE)
add_library(yolo26n-buckyball-pebble-runner SHARED EXCLUDE_FROM_ALL runner.cpp)
set_target_properties(yolo26n-buckyball-pebble-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_include_directories(yolo26n-buckyball-pebble-runner
  PRIVATE ${LIB_DIR}
          ${FLATBUFFERS_INCLUDE}
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/runtime/include
          ${DESIGN_COMPILER}/runtime/include
)
target_link_libraries(yolo26n-buckyball-pebble-runner
  PRIVATE dl
          model_image
)

# ===----------------------------------------------------------------------===
# Package the RAX
#
# Contents:
# RISC-V model library (libyolo26n-buckyball-pebble-model.so),
# RISC-V runner library (libyolo26n-buckyball-pebble-runner.so),
# FP32 parameters, anchors and strides (arg0.data), and the COCO class names
# (labels.txt).
#
# Output: output/yolo26n.rax embeds these files alongside RHAL metadata
# identifying the runner, model entry and parameter buffers.
# ===----------------------------------------------------------------------===
configure_file(yolo26n.rhal.mlir.in ${OUTPUT_DIR}/yolo26n.rhal.mlir @ONLY)
configure_file(${MODEL_DIR}/yolo26n/labels.txt ${OUTPUT_DIR}/labels.txt COPYONLY)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/yolo26n.rax
  COMMAND ${DESIGN_COMPILER}/bin/rax-pack ${OUTPUT_DIR}/yolo26n.rhal.mlir
          --embed-payload
          -o ${OUTPUT_DIR}/yolo26n.rax
  DEPENDS ${OUTPUT_DIR}/yolo26n.rhal.mlir
          ${DESIGN_COMPILER}/bin/rax-pack
          yolo26n-buckyball-pebble-model
          yolo26n-buckyball-pebble-runner
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/labels.txt
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging yolo26n for buckyball-pebble"
  VERBATIM
)
add_custom_target(yolo26n-model-build DEPENDS ${OUTPUT_DIR}/yolo26n.rax)
