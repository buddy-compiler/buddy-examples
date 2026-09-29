set(DIP_SOURCE "${BUDDY_SOURCE}/frontend/Interfaces/lib/DIP.mlir")
add_custom_command(OUTPUT DIP.o
  COMMAND bash -o pipefail -c
    "'${COMPILER_BUILD}/bin/buddy-opt' '${DIP_SOURCE}' -lower-dip -arith-expand -lower-affine -convert-scf-to-cf -convert-math-to-llvm -convert-math-to-libm -convert-vector-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm -reconcile-unrealized-casts | '${COMPILER_BUILD}/bin/buddy-translate' --mlir-to-llvmir | '${COMPILER_BUILD}/bin/buddy-llc' -filetype=obj -mtriple=riscv64 -O2 -mattr=+D -float-abi=hard -o DIP.o"
  DEPENDS ${DIP_SOURCE} ${TOOLS} VERBATIM)
add_library(model_image STATIC DIP.o)
set_target_properties(model_image PROPERTIES LINKER_LANGUAGE CXX)
