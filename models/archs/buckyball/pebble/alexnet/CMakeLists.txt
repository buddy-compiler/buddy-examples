set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")
set(PAYLOAD_DIR "${OUTPUT_DIR}/alexnet.payload")

# ===----------------------------------------------------------------------===
# Import AlexNet into MLIR and export its quantized model data.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/forward.mlir
         ${OUTPUT_DIR}/subgraph0.mlir
         ${OUTPUT_DIR}/model.json
         ${PAYLOAD_DIR}/params.f32
         ${PAYLOAD_DIR}/weights.bin
         ${PAYLOAD_DIR}/scales.bin
         ${PAYLOAD_DIR}/quant-index.json
  COMMAND ${PYTHON} buddy-pytorch-importer.py
            --repo-root ${REPO_ROOT}
            --output-dir ${OUTPUT_DIR}
            --compiler-build ${COMPILER_BUILD}
            --design ${DESIGN}
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          ${REPO_ROOT}/examples/designs/pebble/stack/midend/ops.py
          buddy-pytorch-importer.py
          ${MODEL_DIR}/alexnet/output/alexnet/model.pth
          ${MODEL_DIR}/alexnet/images/dog.bmp
          quant/calibration.py
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
  COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/forward.mlir -buffer-deallocation-simplification -convert-linalg-to-loops -eliminate-empty-tensors -llvm-request-c-wrappers -convert-math-to-llvm -convert-math-to-libm -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -expand-strided-metadata -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts -o ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/forward_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/forward.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/forward.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o forward.o
  DEPENDS "${OUTPUT_DIR}/forward.mlir"
          ${DESIGN_TARGET_PLUGIN_pebble}
          ${DESIGN_COMPILER}/bin/buddy-translate
          ${DESIGN_COMPILER}/bin/buddy-llc
  COMMENT "Building forward.o"
  VERBATIM
)
# ===----------------------------------------------------------------------===
# Lower the compute graphs to RISC-V objects and export intermediate MLIR.
# ===----------------------------------------------------------------------===
set(KERNEL_OBJECTS)
foreach(GRAPH subgraph0)
  add_custom_command(
    OUTPUT ${GRAPH}.o
    BYPRODUCTS ${OUTPUT_DIR}/${GRAPH}.ll
               ${OUTPUT_DIR}/${GRAPH}_linalg.mlir
               ${OUTPUT_DIR}/${GRAPH}_buckyball.mlir
               ${OUTPUT_DIR}/${GRAPH}_bank_ssa.mlir
               ${OUTPUT_DIR}/${GRAPH}_llvm.mlir
    COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/${GRAPH}.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" -o ${OUTPUT_DIR}/${GRAPH}_linalg.mlir
    COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/${GRAPH}_linalg.mlir -eliminate-empty-tensors -convert-elementwise-to-linalg -convert-tensor-to-linalg -one-shot-bufferize=bufferize-function-boundaries -form-pebble-mega-kernels -convert-linalg-to-buckyball -convert-linalg-to-toint8-ball -convert-linalg-to-transpose-ball -o ${OUTPUT_DIR}/${GRAPH}_buckyball.mlir
    COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/${GRAPH}_buckyball.mlir -lower-pebble-to-bank-ssa -assign-physical-banks -o ${OUTPUT_DIR}/${GRAPH}_bank_ssa.mlir
    COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/${GRAPH}_bank_ssa.mlir -llvm-request-c-wrappers -lower-buckyball -convert-trace-to-llvm -expand-strided-metadata -convert-linalg-to-loops -lower-buckyball -canonicalize -cse -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -buffer-deallocation-simplification -bufferization-lower-deallocations -convert-math-to-llvm -convert-math-to-libm -convert-arith-to-llvm -convert-func-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUTPUT_DIR}/${GRAPH}_llvm.mlir
    COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/${GRAPH}_llvm.mlir
              --buddy-to-llvmir -o ${OUTPUT_DIR}/${GRAPH}.ll
    COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/${GRAPH}.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o ${GRAPH}.o
    DEPENDS "${OUTPUT_DIR}/${GRAPH}.mlir"
            ${DESIGN_TARGET_PLUGIN_pebble}
            ${DESIGN_COMPILER}/bin/buddy-translate
            ${DESIGN_COMPILER}/bin/buddy-llc
    COMMENT "Building ${GRAPH}.o"
    VERBATIM
  )
  list(APPEND KERNEL_OBJECTS ${GRAPH}.o)
endforeach()
# ===----------------------------------------------------------------------===
# Link the model and runtime into a RISC-V shared library.
# ===----------------------------------------------------------------------===
add_library(alexnet-buckyball-pebble-model SHARED EXCLUDE_FROM_ALL forward.o ${KERNEL_OBJECTS})
set_target_properties(alexnet-buckyball-pebble-model
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             LINKER_LANGUAGE CXX
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_link_libraries(alexnet-buckyball-pebble-model
  PRIVATE runtime-linux-shared
)

# ===----------------------------------------------------------------------===
# Build the Buddy InferenceRunner plugin for RISC-V Linux.
# ===----------------------------------------------------------------------===
get_filename_component(FLATBUFFERS_HEADER "${TOOLCHAIN}/include/flatbuffers/flatbuffers.h" REALPATH)
get_filename_component(FLATBUFFERS_INCLUDE "${FLATBUFFERS_HEADER}/../.." ABSOLUTE)
add_library(alexnet-buckyball-pebble-runner SHARED EXCLUDE_FROM_ALL runner.cpp)
set_target_properties(alexnet-buckyball-pebble-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
target_include_directories(alexnet-buckyball-pebble-runner
  PRIVATE ${LIB_DIR}
          ${FLATBUFFERS_INCLUDE}
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/runtime/include
          ${DESIGN_COMPILER}/runtime/include
)
target_link_libraries(alexnet-buckyball-pebble-runner
  PRIVATE dl
)

# ===----------------------------------------------------------------------===
# Package the RAX
#
# Contents:
# RISC-V model library (libalexnet-buckyball-pebble-model.so),
# RISC-V runner library (libalexnet-buckyball-pebble-runner.so),
# FP32 parameters (alexnet.payload/params.f32), and
# INT8 weights (alexnet.payload/weights.bin).
#
# Output: output/alexnet.rax embeds these files alongside RHAL metadata
# identifying the runner, model entry and parameter buffers.
# ===----------------------------------------------------------------------===
configure_file(alexnet.rhal.mlir.in ${OUTPUT_DIR}/alexnet.rhal.mlir @ONLY)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/alexnet.rax
  COMMAND ${DESIGN_COMPILER}/bin/rax-pack ${OUTPUT_DIR}/alexnet.rhal.mlir
          --embed-payload
          -o ${OUTPUT_DIR}/alexnet.rax
  DEPENDS ${OUTPUT_DIR}/alexnet.rhal.mlir
          ${DESIGN_COMPILER}/bin/rax-pack
          alexnet-buckyball-pebble-model
          alexnet-buckyball-pebble-runner
          ${OUTPUT_DIR}/alexnet.payload/params.f32
          ${OUTPUT_DIR}/alexnet.payload/weights.bin
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging alexnet for buckyball-pebble"
  VERBATIM
)
add_custom_target(alexnet-model-build DEPENDS ${OUTPUT_DIR}/alexnet.rax)
