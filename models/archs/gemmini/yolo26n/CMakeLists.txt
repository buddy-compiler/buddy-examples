set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

add_custom_command(
  OUTPUT ${OUTPUT_DIR}/forward.mlir
         ${OUTPUT_DIR}/subgraph0.mlir
         ${OUTPUT_DIR}/arg0.data
         ${OUTPUT_DIR}/labels.txt
  COMMAND ${PYTHON} buddy-pytorch-importer.py
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          trace/trace.toml
          ${MODEL_DIR}/yolo26n/output/yolo26n/yolo26n.pt
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing yolo26n for gemmini"
  VERBATIM
)

add_custom_command(
  OUTPUT forward.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${GEMMINI_OPT} ${OUTPUT_DIR}/forward.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${GEMMINI_OPT} ${OUTPUT_DIR}/forward.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${COMPILER_BUILD}/bin/buddy-translate
            --buddy-to-llvmir |
          ${COMPILER_BUILD}/bin/buddy-llc -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+xgemmini,+D -float-abi=hard -o forward.o
  DEPENDS ${OUTPUT_DIR}/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM
  DEPENDS ${GEMMINI_PLUGIN}
)

add_custom_command(
  OUTPUT subgraph0.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_gemmini.mlir
             ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${GEMMINI_OPT} ${OUTPUT_DIR}/subgraph0.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -prepare-gemmini-int8
            -convert-linalg-to-gemmini
            -o ${OUTPUT_DIR}/subgraph0_gemmini.mlir
  COMMAND ${GEMMINI_OPT} ${OUTPUT_DIR}/subgraph0.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -prepare-gemmini-int8
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -lower-gemmini
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${GEMMINI_OPT} ${OUTPUT_DIR}/subgraph0.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -prepare-gemmini-int8
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -lower-gemmini
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${COMPILER_BUILD}/bin/buddy-translate
            --buddy-to-llvmir |
          ${COMPILER_BUILD}/bin/buddy-llc -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+xgemmini,+D -float-abi=hard -o subgraph0.o
  DEPENDS ${OUTPUT_DIR}/subgraph0.mlir
  COMMENT "Building subgraph0.o"
  VERBATIM
  DEPENDS ${GEMMINI_PLUGIN}
)

add_library(yolo26n-gemmini-model STATIC EXCLUDE_FROM_ALL forward.o subgraph0.o)
set_target_properties(yolo26n-gemmini-model
  PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             LINKER_LANGUAGE CXX
)

add_library(yolo26n-gemmini-runner SHARED EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/runner.cpp)
set_target_properties(yolo26n-gemmini-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
)
add_dependencies(yolo26n-gemmini-runner yolo26n-gemmini-model)

target_include_directories(yolo26n-gemmini-runner
  PRIVATE ${BUDDY_SOURCE}/frontend/Interfaces
          ${LIB_DIR}
)
target_link_libraries(yolo26n-gemmini-runner
  PRIVATE yolo26n-gemmini-model
          gemmini-runtime
          model_image
)

# The runner owns inference and the backend's native scheduling lifecycle.
get_filename_component(FLATBUFFERS_HEADER "${TOOLCHAIN}/include/flatbuffers/flatbuffers.h" REALPATH)
get_filename_component(FLATBUFFERS_INCLUDE "${FLATBUFFERS_HEADER}/../.." ABSOLUTE)
target_include_directories(yolo26n-gemmini-runner
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
set_target_properties(yolo26n-gemmini-runner
  PROPERTIES POSITION_INDEPENDENT_CODE ON
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
)
configure_file(yolo26n.rhal.mlir.in ${OUTPUT_DIR}/yolo26n.rhal.mlir @ONLY)

# Package the runner, model parameters and input metadata in one RAX.
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/yolo26n.rax
  COMMAND ${COMPILER_BUILD}/bin/rax-pack ${OUTPUT_DIR}/yolo26n.rhal.mlir
          --embed-payload
          -o ${OUTPUT_DIR}/yolo26n.rax
  DEPENDS ${OUTPUT_DIR}/yolo26n.rhal.mlir
          ${COMPILER_BUILD}/bin/rax-pack
          yolo26n-gemmini-runner
          ${OUTPUT_DIR}/arg0.data
          ${OUTPUT_DIR}/labels.txt
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packaging yolo26n for gemmini"
  VERBATIM
)
add_custom_target(yolo26n-model-build DEPENDS ${OUTPUT_DIR}/yolo26n.rax)
