set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

add_custom_command(
  OUTPUT ${OUTPUT_DIR}/forward.mlir
         ${OUTPUT_DIR}/subgraph0.mlir
         ${OUTPUT_DIR}/arg0.data
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
  COMMAND ${PYTHON} buddy-pytorch-importer.py
            --repo-root ${REPO_ROOT}
            --output-dir ${OUTPUT_DIR}
            --compiler-build ${COMPILER_BUILD}
            --checkpoint ${MODEL_DIR}/lenet/output/lenet-model.pth
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${MODEL_DIR}/lenet/output/lenet-model.pth
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing lenet for gemmini"
  VERBATIM
)

add_custom_command(
  OUTPUT forward.o
  BYPRODUCTS ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -buffer-deallocation-simplification
            -convert-linalg-to-loops
            -eliminate-empty-tensors
            -llvm-request-c-wrappers
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-scf-to-cf
            -convert-arith-to-llvm
            -expand-strided-metadata
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/forward_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/forward.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -buffer-deallocation-simplification
            -convert-linalg-to-loops
            -eliminate-empty-tensors
            -llvm-request-c-wrappers
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-scf-to-cf
            -convert-arith-to-llvm
            -expand-strided-metadata
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${COMPILER_BUILD}/bin/buddy-translate
            --buddy-to-llvmir |
          ${COMPILER_BUILD}/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xgemmini,+D -float-abi=hard -o forward.o
  DEPENDS ${OUTPUT_DIR}/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM
)

add_custom_command(
  OUTPUT subgraph0.o
  BYPRODUCTS ${OUTPUT_DIR}/subgraph0_linalg.mlir
             ${OUTPUT_DIR}/subgraph0_loops.mlir
             ${OUTPUT_DIR}/subgraph0_gemmini.mlir
             ${OUTPUT_DIR}/subgraph0.ll
             ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-gemmini
            -o ${OUTPUT_DIR}/subgraph0_gemmini.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -lower-gemmini
            -convert-trace-to-llvm
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/subgraph0_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))"
            -o ${OUTPUT_DIR}/subgraph0_linalg.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-gemmini
            -batchmatmul-optimize
            -o ${OUTPUT_DIR}/subgraph0_loops.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-gemmini
            -lower-gemmini
            -o ${OUTPUT_DIR}/subgraph0_gemmini.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -lower-gemmini
            -convert-trace-to-llvm
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${COMPILER_BUILD}/bin/buddy-translate
            --buddy-to-llvmir -o ${OUTPUT_DIR}/subgraph0.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${COMPILER_BUILD}/bin/buddy-opt
            -eliminate-empty-tensors
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -lower-gemmini
            -convert-trace-to-llvm
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${COMPILER_BUILD}/bin/buddy-translate
            --buddy-to-llvmir |
          ${COMPILER_BUILD}/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xgemmini,+D -float-abi=hard -o subgraph0.o
  DEPENDS ${OUTPUT_DIR}/subgraph0.mlir
  COMMENT "Building subgraph0.o"
  VERBATIM
)

add_library(lenet-gemmini-model STATIC EXCLUDE_FROM_ALL subgraph0.o forward.o)
set_target_properties(lenet-gemmini-model PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
set_target_properties(lenet-gemmini-model PROPERTIES LINKER_LANGUAGE CXX)

add_executable(buddy-gemmini-lenet-run EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
set_target_properties(buddy-gemmini-lenet-run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
add_dependencies(buddy-gemmini-lenet-run lenet-gemmini-model)
target_include_directories(buddy-gemmini-lenet-run PRIVATE ${BUDDY_SOURCE}/frontend/Interfaces
                                                           ${MODEL_LIB_DIR}
)
target_link_libraries(buddy-gemmini-lenet-run -static lenet-gemmini-model
                                                      gemmini-runtime
)

add_custom_target(lenet-model-build DEPENDS lenet-gemmini-model)
