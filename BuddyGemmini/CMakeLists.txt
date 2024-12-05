set(BUDDY_EXAMPLES_DIR ${BUDDY_MLIR_DIR}/examples/)
set(BUDDY_BINARY_DIR ${BUDDY_MLIR_DIR}/build/bin/)
set(RISCV_GNU_TOOLCHAIN ${BUDDY_MLIR_DIR}/build/thirdparty/riscv-gnu-toolchain)
set(CMAKE_CXX_COMPILER  ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

set(BUDDY_GEMMINI_DIR ${BUDDY_EXAMPLES_DIR}/BuddyGemmini)
set(INTERFACE_DIR ${BUDDY_MLIR_DIR}/frontend/Interfaces/)
set(INCLUDE_DIR ${BUDDY_GEMMINI_DIR}/include/)


add_custom_command(
  OUTPUT  forward.mlir subgraph0.mlir arg0.data
  COMMAND python3 ${BUDDY_GEMMINI_DIR}/buddy-lenet-import.py
  DEPENDS buddy-lenet-import.py
  COMMENT "Generating forward.mlir, subgraph0.mlir and parameter files"
)

add_custom_command(
  OUTPUT  forward.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${BUDDY_GEMMINI_DIR}/forward.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg, arith-bufferize, func.func(linalg-bufferize, tensor-bufferize), func-bufferize)" |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -buffer-deallocation-simplification
            -convert-linalg-to-loops
            -eliminate-empty-tensors
            -llvm-request-c-wrappers
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-scf-to-cf
            -lower-gemmini
            -convert-arith-to-llvm
            -expand-strided-metadata
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O0 -mattr=+buddyext,+D -float-abi=hard -o forward.o
          DEPENDS forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)

add_custom_command(
  OUTPUT  subgraph0.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${BUDDY_GEMMINI_DIR}/subgraph0.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -convert-tensor-to-linalg
            -linalg-bufferize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -lower-affine
            -func-bufferize-dynamic-offset
            -arith-bufferize
            -tensor-bufferize
            -buffer-deallocation
            -finalizing-bufferize
            -convert-vector-to-scf
            -expand-strided-metadata
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -lower-gemmini 
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts | 
          ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O0 -mattr=+buddyext,+D -float-abi=hard -o subgraph0.o
          DEPENDS subgraph0.mlir
  COMMENT "Building subgraph0.o"
  VERBATIM)


add_library(GemminiLENET STATIC subgraph0.o forward.o)
set_target_properties(GemminiLENET PROPERTIES LINKER_LANGUAGE CXX)

add_executable(buddy-gemmini-lenet-run buddy-lenet-main.cpp)
add_dependencies(buddy-gemmini-lenet-run GemminiLENET)
target_include_directories(buddy-gemmini-lenet-run PRIVATE ${INTERFACE_DIR} ${INCLUDE_DIR})
target_link_libraries(buddy-gemmini-lenet-run -static GemminiLENET)