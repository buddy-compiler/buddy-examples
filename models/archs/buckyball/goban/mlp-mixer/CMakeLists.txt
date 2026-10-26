file(GLOB_RECURSE RECIPE_FRONTEND_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/frontend/*.py")

set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")
# Model parameters and quantization metadata under OUTPUT_DIR.
set(PAYLOAD_DIR "${OUTPUT_DIR}/mixer.payload")
# Model parameters and quantization metadata under OUTPUT_DIR.
set(PAYLOAD_DIR "${PAYLOAD_DIR}")

set(DESIGN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/trace/trace.toml")
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/forward.mlir
         ${OUTPUT_DIR}/subgraph0.mlir
         ${OUTPUT_DIR}/model.json
         ${OUTPUT_DIR}/input.f32
         ${OUTPUT_DIR}/arg0.data
         ${OUTPUT_DIR}/arg1.data
         ${PAYLOAD_DIR}/scales.bin
         ${PAYLOAD_DIR}/quant-index.json
  COMMAND ${PYTHON} buddy-pytorch-importer.py
            --repo-root ${REPO_ROOT}
            --checkpoint "timm/mixer_b16_224.goog_in21k_ft_in1k"
            --revision "0006300d604941f96f16cb7d229137363e1b2ac6"
            --output-dir ${OUTPUT_DIR}
            --compiler-build ${COMPILER_BUILD}
            --design-pb ${DESIGN_PB}
            --design ${DESIGN}
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          ${RECIPE_FRONTEND_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          quant/calibration.py
          quant/quant.toml
          ${MODEL_DIR}/mlp-mixer/model.py
          ${MODEL_DIR}/mlp-mixer/../alexnet/images/dog.bmp
          ${DESIGN_FILES}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)
add_custom_command(
  OUTPUT forward.o
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUTPUT_DIR}/forward.mlir -buffer-deallocation-simplification -convert-linalg-to-loops -eliminate-empty-tensors -llvm-request-c-wrappers -convert-math-to-llvm -convert-math-to-libm -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -expand-strided-metadata -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts -o ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/forward_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/forward.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/forward.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o forward.o
  COMMAND ${CMAKE_OBJCOPY}
            --redefine-sym malloc=workspace_alloc
            --redefine-sym free=workspace_free forward.o
  DEPENDS ${OUTPUT_DIR}/forward.mlir
          ${DESIGN_TARGET_PLUGIN_matmul}
  VERBATIM
)
add_custom_command(
  OUTPUT subgraph0.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_buckyball.mlir
             ${OUTPUT_DIR}/subgraph0_bank_ssa.mlir
             ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUTPUT_DIR}/subgraph0.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" -o ${OUTPUT_DIR}/subgraph0_linalg.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUTPUT_DIR}/subgraph0_linalg.mlir -eliminate-empty-tensors -convert-elementwise-to-linalg -convert-tensor-to-linalg -one-shot-bufferize=bufferize-function-boundaries -form-goban-mega-kernels -convert-linalg-to-buckyball -convert-linalg-to-toint8-ball -o ${OUTPUT_DIR}/subgraph0_buckyball.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUTPUT_DIR}/subgraph0_buckyball.mlir -lower-matmul-region-to-bank-ssa -lower-buckyball-to-bank-ssa -assign-physical-banks -o ${OUTPUT_DIR}/subgraph0_bank_ssa.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUTPUT_DIR}/subgraph0_bank_ssa.mlir -llvm-request-c-wrappers -lower-buckyball -convert-trace-to-llvm -expand-strided-metadata -convert-linalg-to-loops -lower-buckyball -canonicalize -cse -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -buffer-deallocation-pipeline -buffer-deallocation-simplification -bufferization-lower-deallocations -convert-math-to-llvm -convert-math-to-libm -convert-arith-to-llvm -convert-func-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts -lower-to-stream -o ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/subgraph0_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/subgraph0.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/subgraph0.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o subgraph0.o
  COMMAND ${CMAKE_OBJCOPY}
            --redefine-syms=${DESIGN_THREAD_BUILD_matmul}/kernels.rename
            --redefine-sym malloc=workspace_alloc
            --redefine-sym free=workspace_free subgraph0.o
  DEPENDS ${OUTPUT_DIR}/subgraph0.mlir
          ${DESIGN_TARGET_PLUGIN_matmul}
          ${DESIGN_THREAD_BUILD_matmul}/kernels.rename
  VERBATIM
)
add_library(mlp-mixer-buckyball-goban-model STATIC EXCLUDE_FROM_ALL forward.o subgraph0.o)
set_target_properties(mlp-mixer-buckyball-goban-model
  PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             LINKER_LANGUAGE CXX
)

add_library(mlp-mixer-buckyball-goban-runner SHARED EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/runner.cpp)
set_target_properties(mlp-mixer-buckyball-goban-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
)
target_include_directories(mlp-mixer-buckyball-goban-runner
  PRIVATE ${OUTPUT_DIR}
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/llvm/mlir/include
          ${LIB_DIR}
)
target_include_directories(mlp-mixer-buckyball-goban-runner
  PRIVATE ${RUNTIME_PARAMS}
)
target_link_libraries(mlp-mixer-buckyball-goban-runner
  PRIVATE mlp-mixer-buckyball-goban-model
          runtime-linux
)

# The runner owns inference and the backend's native scheduling lifecycle.
get_filename_component(FLATBUFFERS_HEADER "${TOOLCHAIN}/include/flatbuffers/flatbuffers.h" REALPATH)
get_filename_component(FLATBUFFERS_INCLUDE "${FLATBUFFERS_HEADER}/../.." ABSOLUTE)
target_include_directories(mlp-mixer-buckyball-goban-runner
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
set_target_properties(mlp-mixer-buckyball-goban-runner
  PROPERTIES POSITION_INDEPENDENT_CODE ON
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
configure_file(mlp-mixer.rhal.mlir.in ${OUTPUT_DIR}/mlp-mixer.rhal.mlir @ONLY)

# Package the runner, model parameters and input metadata in one RAX.
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/mlp-mixer.rax
  COMMAND ${COMPILER_BUILD}/bin/rax-pack ${OUTPUT_DIR}/mlp-mixer.rhal.mlir
          --embed-payload
          -o ${OUTPUT_DIR}/mlp-mixer.rax
  DEPENDS ${OUTPUT_DIR}/mlp-mixer.rhal.mlir
          ${COMPILER_BUILD}/bin/rax-pack
          mlp-mixer-buckyball-goban-runner
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging mlp-mixer for buckyball-goban"
  VERBATIM
)
add_custom_target(mlp-mixer-model-build DEPENDS ${OUTPUT_DIR}/mlp-mixer.rax)
