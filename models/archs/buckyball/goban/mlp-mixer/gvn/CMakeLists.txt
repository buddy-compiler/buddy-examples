# The GVN slices of mlp-mixer, in gvn/mlp-mixer of the config output: the hart graph,
# its thread subgraphs checked at every DDR result; then (configured again
# once the checks exist) one slice per check of each subgraph, cut into
# batches of one tile iteration, lowered like the recipe's graph. The whole
# graph is lowered to its physical banks, which every slice must follow, and
# the operators of the checked graph must equal the compiled model's. Each
# slice's ELF, with its readback, is goban's (gvn/ of its device simulation).

# The checks, by the GVN passes; locations kept for matching the GPU's values.
add_custom_command(
  OUTPUT ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/checks.mlir
         ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/gvn_operators.mlir
         ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/model_operators.mlir
  BYPRODUCTS ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/hart.mlir
             ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/acc.mlir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu
  COMMAND ${DESIGN_HART_OPT_goban}
            --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so -mlir-print-debuginfo ${OUTPUT_DIR}/subgraph0_linalg.mlir -gvn-source-origins -convert-elementwise-to-linalg -convert-linalg-to-hart=design=${DESIGN_PB} -o ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul}
            --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            -mlir-print-debuginfo ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/hart.mlir
            -convert-linalg-to-acc
            -o ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/acc.mlir
  COMMAND ${DESIGN_HART_OPT_goban}
            --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so -mlir-print-debuginfo ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/acc.mlir -gvn-insert-checks -o ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/checks.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/acc.mlir -symbol-privatize=exclude=operator,gp_space -symbol-dce -canonicalize -strip-debuginfo -o ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/gvn_operators.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUTPUT_DIR}/subgraph0_acc.mlir -symbol-privatize=exclude=operator,gp_space -symbol-dce -canonicalize -strip-debuginfo -o ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/model_operators.mlir
  DEPENDS ${OUTPUT_DIR}/subgraph0_linalg.mlir
          ${OUTPUT_DIR}/subgraph0_acc.mlir
          ${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
          ${DESIGN_HART_PLUGIN_goban}
          ${DESIGN_TARGET_PLUGIN_matmul}
          ${DESIGN_PB}
  COMMAND_EXPAND_LISTS
  VERBATIM
)
add_custom_target(
  mlp-mixer-gvn-checks
  DEPENDS
    ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/checks.mlir
    ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/gvn_operators.mlir
    ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/model_operators.mlir
)

# The whole graph, to its physical banks.
add_custom_command(
  OUTPUT ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/whole_physical.mlir
  BYPRODUCTS ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/whole_frontend.mlir
             ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/whole_bufferized.mlir
             ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/whole_cwrappers.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/acc.mlir -lower-frontend-fake-op -o ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/whole_frontend.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/whole_frontend.mlir -eliminate-empty-tensors -convert-elementwise-to-linalg -convert-tensor-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -drop-equivalent-buffer-results -buffer-results-to-out-params=hoist-static-allocs -ownership-based-buffer-deallocation -buffer-deallocation-simplification -bufferization-lower-deallocations -canonicalize -o ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/whole_bufferized.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/whole_bufferized.mlir -llvm-request-c-wrappers -o ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/whole_cwrappers.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/whole_cwrappers.mlir
            -assign-physical-banks
            -o ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/whole_physical.mlir
  DEPENDS ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/acc.mlir
          ${DESIGN_HART_PLUGIN_goban}
          ${DESIGN_TARGET_PLUGIN_matmul}
          ${DESIGN_PB}
  COMMAND_EXPAND_LISTS
  VERBATIM
)

# One slice per check of each thread subgraph, once the checks exist: a
# check belongs to the function it is in.
set(objects)
if(EXISTS ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/checks.mlir)
  file(STRINGS ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/checks.mlir lines REGEX "func\\.func |gvn\\.check ")
  foreach(line ${lines})
    if(line MATCHES "func\\.func (private )?@([A-Za-z0-9_]+)\\(")
      set(function ${CMAKE_MATCH_2})
      set(index 0)
    else()
      set(slice ${DESIGN_OUTPUT}/gvn/mlp-mixer/${function}.${index})
      add_custom_command(
        OUTPUT ${slice}/slice.o
               ${slice}/slice_physical.mlir
               ${slice}/layout.json
        COMMAND ${CMAKE_COMMAND} -E make_directory ${slice}
        COMMAND ${DESIGN_HART_OPT_goban}
                  --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
                  --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so -mlir-print-debuginfo ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/checks.mlir -canonicalize "-gvn-slice=function=${function} index=${index} iterate=true batch=1 layout=${slice}/layout.json" -o ${slice}/slice.mlir
        COMMAND ${DESIGN_HART_OPT_goban} ${slice}/slice.mlir -lower-frontend-fake-op -o ${slice}/slice_frontend.mlir
        COMMAND ${DESIGN_HART_OPT_goban} ${slice}/slice_frontend.mlir -eliminate-empty-tensors -convert-elementwise-to-linalg -convert-tensor-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -drop-equivalent-buffer-results -buffer-results-to-out-params=hoist-static-allocs -ownership-based-buffer-deallocation -buffer-deallocation-simplification -bufferization-lower-deallocations -canonicalize -o ${slice}/slice_bufferized.mlir
        COMMAND ${DESIGN_HART_OPT_goban} ${slice}/slice_bufferized.mlir -llvm-request-c-wrappers -o ${slice}/slice_cwrappers.mlir
        COMMAND ${DESIGN_TARGET_OPT_matmul} ${slice}/slice_cwrappers.mlir
                  -assign-physical-banks
                  -o ${slice}/slice_physical.mlir
        COMMAND ${DESIGN_TARGET_OPT_matmul} ${slice}/slice_physical.mlir
                  -lower-acc
                  -o ${slice}/slice_lowered.mlir
        COMMAND ${DESIGN_HART_OPT_goban} ${slice}/slice_lowered.mlir -expand-strided-metadata -lower-affine -canonicalize -o ${slice}/slice_addresses.mlir
        COMMAND ${DESIGN_HART_OPT_goban} ${slice}/slice_addresses.mlir "-lower-hart=design=${DESIGN_PB} hart=goban" -o ${slice}/slice_memref.mlir
        COMMAND ${DESIGN_HART_OPT_goban} ${slice}/slice_memref.mlir -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${slice}/slice_copies.mlir
        COMMAND ${DESIGN_TARGET_OPT_matmul} ${slice}/slice_copies.mlir
                  -symbol-privatize=exclude=gvn_entry,gvn_in,gvn_out,gvn_table,_binary_operators_image_start
                  -symbol-dce
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
                  -finalize-memref-to-llvm=use-aligned-alloc
                  -reconcile-unrealized-casts
                  -o ${slice}/slice_llvm.mlir
        COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${slice}/slice_llvm.mlir
                  --buddy-to-llvmir -o ${slice}/slice.ll
        COMMAND ${LLVM_OPT} ${slice}/slice.ll -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=default<O2>" -S -o ${slice}/slice_opt.ll
        COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${slice}/slice_opt.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -max-store-memcpy=16 -mattr=+xbuckyball,+m,+D -float-abi=hard -o ${slice}/slice.o
        # Each intermediate holds the whole module (up to 80 MB).
        COMMAND ${CMAKE_COMMAND} -E rm -f ${slice}/slice.mlir ${slice}/slice_frontend.mlir ${slice}/slice_bufferized.mlir ${slice}/slice_cwrappers.mlir ${slice}/slice_lowered.mlir ${slice}/slice_addresses.mlir ${slice}/slice_memref.mlir ${slice}/slice_copies.mlir ${slice}/slice_llvm.mlir ${slice}/slice.ll ${slice}/slice_opt.ll
        DEPENDS ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/checks.mlir
                ${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
                ${DESIGN_HART_PLUGIN_goban}
                ${DESIGN_TARGET_PLUGIN_matmul}
                ${DESIGN_PB}
        COMMAND_EXPAND_LISTS
        VERBATIM
      )
      list(APPEND objects ${slice}/slice.o)
      math(EXPR index "${index} + 1")
    endif()
  endforeach()
endif()
# The model's operators, for the slice ELFs.
add_custom_command(
  OUTPUT ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/operators.embed.o
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/../operators.embed.o ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/operators.embed.o
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/../operators.embed.o
  VERBATIM
)
add_custom_target(
  mlp-mixer-gvn
  DEPENDS
    ${objects}
    ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/whole_physical.mlir
    ${DESIGN_OUTPUT}/gvn/mlp-mixer/npu/operators.embed.o
)
add_dependencies(mlp-mixer-gvn mlp-mixer-model-build)
