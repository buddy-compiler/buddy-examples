set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")
# Model parameters and quantization metadata under OUTPUT_DIR.
set(PAYLOAD_DIR "${OUTPUT_DIR}/resnet18.payload")

set(DESIGN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/trace/trace.toml")
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/forward.mlir
         ${OUTPUT_DIR}/subgraph0.mlir
         ${OUTPUT_DIR}/model.json
         ${OUTPUT_DIR}/Labels.txt
         ${OUTPUT_DIR}/arg0.data
         ${OUTPUT_DIR}/arg1.data
         ${PAYLOAD_DIR}/scales.bin
         ${PAYLOAD_DIR}/quant-index.json
  BYPRODUCTS ${OUTPUT_DIR}/activation-scales.json
  COMMAND ${PYTHON} buddy-pytorch-importer.py
            --repo-root ${REPO_ROOT}
            --output-dir ${OUTPUT_DIR}
            --compiler-build ${COMPILER_BUILD}
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          quant/calibration.py
          quant/quant.toml
          ${MODEL_DIR}/resnet18/output/resnet18/model.pth
          ${DESIGN_FILES}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)

add_custom_command(
  OUTPUT forward.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/forward.mlir -buffer-deallocation-simplification -convert-linalg-to-loops -eliminate-empty-tensors -llvm-request-c-wrappers -convert-math-to-llvm -convert-math-to-libm -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -expand-strided-metadata -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts -o ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/forward.mlir -buffer-deallocation-simplification -convert-linalg-to-loops -eliminate-empty-tensors -llvm-request-c-wrappers -convert-math-to-llvm -convert-math-to-libm -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -expand-strided-metadata -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts |
          ${DESIGN_COMPILER}/bin/buddy-translate
            --buddy-to-llvmir |
          ${DESIGN_COMPILER}/bin/buddy-llc -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o forward.o
  DEPENDS "${OUTPUT_DIR}/forward.mlir"
          ${DESIGN_TARGET_PLUGIN_pebble}
          ${DESIGN_COMPILER}/bin/buddy-translate
          ${DESIGN_COMPILER}/bin/buddy-llc
  COMMENT "Building forward.o"
  VERBATIM
)

add_custom_command(
  OUTPUT subgraph0.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_buckyball.mlir
             ${OUTPUT_DIR}/subgraph0_bank_ssa.mlir
             ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/subgraph0.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" -o ${OUTPUT_DIR}/subgraph0_linalg.mlir
  COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/subgraph0.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
          ${DESIGN_TARGET_OPT_pebble} -eliminate-empty-tensors -convert-elementwise-to-linalg -convert-tensor-to-linalg -one-shot-bufferize=bufferize-function-boundaries -form-pebble-mega-kernels -convert-linalg-to-buckyball -convert-linalg-to-toint8-ball -convert-linalg-to-transpose-ball -o ${OUTPUT_DIR}/subgraph0_buckyball.mlir
  COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/subgraph0.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
          ${DESIGN_TARGET_OPT_pebble} -eliminate-empty-tensors -convert-elementwise-to-linalg -convert-tensor-to-linalg -one-shot-bufferize=bufferize-function-boundaries -form-pebble-mega-kernels -convert-linalg-to-buckyball -convert-linalg-to-toint8-ball -convert-linalg-to-transpose-ball -lower-pebble-to-bank-ssa -assign-physical-banks -o ${OUTPUT_DIR}/subgraph0_bank_ssa.mlir
  COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/subgraph0.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
          ${DESIGN_TARGET_OPT_pebble} -eliminate-empty-tensors -convert-elementwise-to-linalg -convert-tensor-to-linalg -one-shot-bufferize=bufferize-function-boundaries -form-pebble-mega-kernels -convert-linalg-to-buckyball -convert-linalg-to-toint8-ball -convert-linalg-to-transpose-ball -lower-pebble-to-bank-ssa -assign-physical-banks -llvm-request-c-wrappers |
          ${DESIGN_TARGET_OPT_pebble} -lower-buckyball -convert-trace-to-llvm -expand-strided-metadata -convert-linalg-to-loops -lower-buckyball -canonicalize -cse -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -buffer-deallocation-simplification -bufferization-lower-deallocations -convert-math-to-llvm -convert-math-to-libm -convert-arith-to-llvm -convert-func-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${DESIGN_TARGET_OPT_pebble} ${OUTPUT_DIR}/subgraph0.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
          ${DESIGN_TARGET_OPT_pebble} -eliminate-empty-tensors -convert-elementwise-to-linalg -convert-tensor-to-linalg -one-shot-bufferize=bufferize-function-boundaries -form-pebble-mega-kernels -convert-linalg-to-buckyball -convert-linalg-to-toint8-ball -convert-linalg-to-transpose-ball -lower-pebble-to-bank-ssa -assign-physical-banks -llvm-request-c-wrappers |
          ${DESIGN_TARGET_OPT_pebble} -lower-buckyball -convert-trace-to-llvm -expand-strided-metadata -convert-linalg-to-loops -lower-buckyball -canonicalize -cse -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -buffer-deallocation-simplification -bufferization-lower-deallocations -convert-math-to-llvm -convert-math-to-libm -convert-arith-to-llvm -convert-func-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts |
          ${DESIGN_COMPILER}/bin/buddy-translate
            --buddy-to-llvmir |
          ${DESIGN_COMPILER}/bin/buddy-llc -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o subgraph0.o
  DEPENDS ${DESIGN_TARGET_PLUGIN_pebble}
          ${DESIGN_COMPILER}/bin/buddy-translate
          "${OUTPUT_DIR}/subgraph0.mlir"
          ${DESIGN_COMPILER}/bin/buddy-llc
  COMMENT "Building subgraph0.o"
  VERBATIM
)
add_library(resnet18-buckyball-pebble-model STATIC EXCLUDE_FROM_ALL forward.o subgraph0.o)
set_target_properties(resnet18-buckyball-pebble-model
  PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             LINKER_LANGUAGE CXX
)

add_library(resnet18-buckyball-pebble-runner SHARED EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/runner.cpp)
set_target_properties(resnet18-buckyball-pebble-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
)
target_include_directories(resnet18-buckyball-pebble-runner
  PRIVATE ${OUTPUT_DIR}
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${LIB_DIR}
)
target_link_libraries(resnet18-buckyball-pebble-runner
  PRIVATE resnet18-buckyball-pebble-model
          runtime-linux
          model_image
)

# The runner owns inference and the backend's native scheduling lifecycle.
get_filename_component(FLATBUFFERS_HEADER "${TOOLCHAIN}/include/flatbuffers/flatbuffers.h" REALPATH)
get_filename_component(FLATBUFFERS_INCLUDE "${FLATBUFFERS_HEADER}/../.." ABSOLUTE)
target_include_directories(resnet18-buckyball-pebble-runner
  PRIVATE ${LIB_DIR}
          ${FLATBUFFERS_INCLUDE}
          ${BUDDY_SOURCE}/runtime/include
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/llvm/mlir/include
          ${OUTPUT_DIR}
          ${ISA_DIR}/design
          ${ISA_DIR}/matmul
          ${RUNTIME_DIR}/scheduler
          ${RUNTIME_DIR}/services
          ${RUNTIME_DIR}/../engine/include
)
set_target_properties(resnet18-buckyball-pebble-runner
  PROPERTIES POSITION_INDEPENDENT_CODE ON
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
configure_file(resnet18.rhal.mlir.in ${OUTPUT_DIR}/resnet18.rhal.mlir @ONLY)

# Package the runner, model parameters and input metadata in one RAX.
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/resnet18.rax
  COMMAND ${COMPILER_BUILD}/bin/rax-pack ${OUTPUT_DIR}/resnet18.rhal.mlir
          --embed-payload
          -o ${OUTPUT_DIR}/resnet18.rax
  DEPENDS ${OUTPUT_DIR}/resnet18.rhal.mlir
          ${COMPILER_BUILD}/bin/rax-pack
          resnet18-buckyball-pebble-runner
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
          ${OUTPUT_DIR}/Labels.txt
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging resnet18 for buckyball-pebble"
  VERBATIM
)
add_custom_target(resnet18-model-build DEPENDS ${OUTPUT_DIR}/resnet18.rax)
