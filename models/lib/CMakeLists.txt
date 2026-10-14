add_custom_command(
  OUTPUT DIP.o
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${BUDDY_SOURCE}/frontend/Interfaces/lib/DIP.mlir
            -lower-dip
            -arith-expand
            -lower-affine
            -convert-scf-to-cf
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-vector-to-llvm
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -convert-cf-to-llvm
            -convert-arith-to-llvm
            -reconcile-unrealized-casts |
          ${COMPILER_BUILD}/bin/buddy-translate
            --mlir-to-llvmir |
          ${BUDDY_SOURCE}/llvm/build/bin/opt "-passes=function(sroa,early-cse,instcombine,simplifycfg,adce)" |
          ${COMPILER_BUILD}/bin/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+D -float-abi=hard -o DIP.o
  DEPENDS ${BUDDY_SOURCE}/frontend/Interfaces/lib/DIP.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
          ${BUDDY_SOURCE}/llvm/build/bin/opt
  VERBATIM
)
add_library(model_image STATIC DIP.o)
set_target_properties(model_image PROPERTIES LINKER_LANGUAGE CXX)
