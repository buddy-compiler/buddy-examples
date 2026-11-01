file(GLOB_RECURSE RECIPE_FRONTEND_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/frontend/*.py")
set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")
# The compiler of the goban core: the core dialect, its placement and
# lowering, and lower-frontend-fake-op; built by the core stack. The graph is placed
# and bufferized with it, then lowered with the matmul thread's compiler.
set(CORE_COMPILER "${DESIGN_CORE_BUILD_goban}/libcore-compiler.so")
set(
  CORE_OPT
  ${DESIGN_COMPILER}/bin/buddy-opt
  --load-dialect-plugin=${CORE_COMPILER}
  --load-pass-plugin=${CORE_COMPILER}
)
set(ACC_OPT ${CORE_OPT} --buckyball-design-pb=${DESIGN_PB} --buckyball-target=matmul)

# ===----------------------------------------------------------------------===
# Calibrate the Linear input and attention operand scales on emotion training
# sentences; rejected when the validation accuracy drops more than 1 point.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/calibration.json
  COMMAND ${PYTHON} buddy-pytorch-calibrate.py
  DEPENDS buddy-pytorch-calibrate.py
          ${STACK_ROOT}/compiler/frontend/quant.py
          ${MODEL_DIR}/bert-base-uncased-emotion/model.py
          ${MODEL_DIR}/bert-base-uncased-emotion/data.py
          ${MODEL_DIR}/bert-base-uncased-emotion/output/bert-base-uncased-emotion/config.json
          ${MODEL_DIR}/bert-base-uncased-emotion/output/bert-base-uncased-emotion/model.safetensors
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Calibrating BERT activations"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Import the BERT graph, permute, quantize every matrix product to INT8 and
# partition it.
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
         ${OUTPUT_DIR}/vocab.txt
  COMMAND ${PYTHON} buddy-pytorch-importer.py
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          ${RECIPE_FRONTEND_SOURCES}
          buddy-pytorch-importer.py
          trace/trace.toml
          ${OUTPUT_DIR}/calibration.json
          ${MODEL_DIR}/bert-base-uncased-emotion/model.py
          ${MODEL_DIR}/bert-base-uncased-emotion/output/bert-base-uncased-emotion/config.json
          ${MODEL_DIR}/bert-base-uncased-emotion/output/bert-base-uncased-emotion/model.safetensors
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing BERT"
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
          ${DESIGN_PB}
          ${DESIGN_COMPILER}/bin/buddy-translate
          ${DESIGN_COMPILER}/bin/buddy-llc
  COMMENT "Building forward.o"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Lower the compute graph and export each compiler stage. The core pass
# outlines thread subgraphs; the acc pass maps their INT8 products,
# quantization and bias decoding to Balls. After bufferization, bank assignment
# and lower-acc produce instructions, which lower-core records as asynchronous
# engine jobs. Host packing finishes before submission and staging allocations
# survive until completion. Request the graph's C interface before lower-core
# so engine entry points keep their C names.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT subgraph0.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0.ll
             ${OUTPUT_DIR}/subgraph0_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_core.mlir
             ${OUTPUT_DIR}/subgraph0_acc.mlir
             ${OUTPUT_DIR}/subgraph0_addresses.mlir
             ${OUTPUT_DIR}/subgraph0_copies.mlir
             ${OUTPUT_DIR}/subgraph0_frontend.mlir
             ${OUTPUT_DIR}/subgraph0_bufferized.mlir
             ${OUTPUT_DIR}/subgraph0_memref.mlir
             ${OUTPUT_DIR}/subgraph0_cwrappers.mlir
             ${OUTPUT_DIR}/subgraph0_physical.mlir
             ${OUTPUT_DIR}/subgraph0_lowered.mlir
             ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/subgraph0.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" -o ${OUTPUT_DIR}/subgraph0_linalg.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/subgraph0_linalg.mlir -convert-linalg-to-core=design=${DESIGN_PB} -o ${OUTPUT_DIR}/subgraph0_core.mlir
  COMMAND ${ACC_OPT} ${OUTPUT_DIR}/subgraph0_core.mlir -convert-linalg-to-acc -o ${OUTPUT_DIR}/subgraph0_acc.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/subgraph0_acc.mlir -lower-frontend-fake-op -o ${OUTPUT_DIR}/subgraph0_frontend.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/subgraph0_frontend.mlir -eliminate-empty-tensors -convert-elementwise-to-linalg -convert-tensor-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -drop-equivalent-buffer-results -buffer-results-to-out-params=hoist-static-allocs -ownership-based-buffer-deallocation -buffer-deallocation-simplification -bufferization-lower-deallocations -canonicalize -o ${OUTPUT_DIR}/subgraph0_bufferized.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/subgraph0_bufferized.mlir -llvm-request-c-wrappers -o ${OUTPUT_DIR}/subgraph0_cwrappers.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUTPUT_DIR}/subgraph0_cwrappers.mlir
            -assign-physical-banks
            -o ${OUTPUT_DIR}/subgraph0_physical.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUTPUT_DIR}/subgraph0_physical.mlir
            -lower-acc
            -o ${OUTPUT_DIR}/subgraph0_lowered.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/subgraph0_lowered.mlir -expand-strided-metadata -lower-affine -canonicalize -o ${OUTPUT_DIR}/subgraph0_addresses.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/subgraph0_addresses.mlir "-lower-core=design=${DESIGN_PB} core=goban" -o ${OUTPUT_DIR}/subgraph0_memref.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/subgraph0_memref.mlir -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUTPUT_DIR}/subgraph0_copies.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUTPUT_DIR}/subgraph0_copies.mlir
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
          ${CORE_COMPILER}
          ${DESIGN_PB}
          ${DESIGN_TARGET_PLUGIN_matmul}
          ${DESIGN_COMPILER}/bin/buddy-translate
          ${DESIGN_COMPILER}/bin/buddy-llc
  COMMENT "Building subgraph0.o"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Link the model and runtime into a RISC-V shared library.
# ===----------------------------------------------------------------------===
add_library(bert-base-uncased-emotion-buckyball-goban-model SHARED EXCLUDE_FROM_ALL forward.o subgraph0.o)
set_target_properties(bert-base-uncased-emotion-buckyball-goban-model
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             LINKER_LANGUAGE CXX
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN;/lib"
)
target_link_libraries(bert-base-uncased-emotion-buckyball-goban-model
  PRIVATE runtime-linux-shared
)

# ===----------------------------------------------------------------------===
# Build the Buddy InferenceRunner plugin for RISC-V Linux.
# ===----------------------------------------------------------------------===
get_filename_component(FLATBUFFERS_HEADER "${TOOLCHAIN}/include/flatbuffers/flatbuffers.h" REALPATH)
get_filename_component(FLATBUFFERS_INCLUDE "${FLATBUFFERS_HEADER}/../.." ABSOLUTE)
add_library(bert-base-uncased-emotion-buckyball-goban-runner SHARED EXCLUDE_FROM_ALL runner.cpp)
set_target_properties(bert-base-uncased-emotion-buckyball-goban-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN;/lib"
)
target_include_directories(bert-base-uncased-emotion-buckyball-goban-runner
  PRIVATE ${LIB_DIR}
          ${FLATBUFFERS_INCLUDE}
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/runtime/include
          ${DESIGN_COMPILER}/runtime/include
)
target_link_libraries(bert-base-uncased-emotion-buckyball-goban-runner
  PRIVATE dl
)

# ===----------------------------------------------------------------------===
# Package the RAX
#
# Contents:
# RISC-V model library (libbert-base-uncased-emotion-buckyball-goban-model.so),
# RISC-V runner library (libbert-base-uncased-emotion-buckyball-goban-runner.so),
# FP32 parameters (arg0.data) and the tokenizer vocabulary (vocab.txt).
#
# Output: output/bert-base-uncased-emotion.rax embeds these files alongside
# RHAL metadata identifying the runner, model entry and parameter buffers.
# ===----------------------------------------------------------------------===
configure_file(bert-base-uncased-emotion.rhal.mlir.in ${OUTPUT_DIR}/bert-base-uncased-emotion.rhal.mlir @ONLY)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/bert-base-uncased-emotion.rax
  COMMAND ${DESIGN_COMPILER}/bin/rax-pack ${OUTPUT_DIR}/bert-base-uncased-emotion.rhal.mlir
          --embed-payload
          -o ${OUTPUT_DIR}/bert-base-uncased-emotion.rax
  DEPENDS ${OUTPUT_DIR}/bert-base-uncased-emotion.rhal.mlir
          ${DESIGN_COMPILER}/bin/rax-pack
          bert-base-uncased-emotion-buckyball-goban-model
          bert-base-uncased-emotion-buckyball-goban-runner
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/arg1.data
          ${OUTPUT_DIR}/arg2.data
          ${OUTPUT_DIR}/vocab.txt
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging bert-base-uncased-emotion for buckyball-goban"
  VERBATIM
)
add_custom_target(bert-base-uncased-emotion-model-build DEPENDS ${OUTPUT_DIR}/bert-base-uncased-emotion.rax)
