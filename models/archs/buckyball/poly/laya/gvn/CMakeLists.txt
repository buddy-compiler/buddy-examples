# The GVN slices of laya, in gvn/laya of the config output, with
# the late hart partition of the CPU operations the frontend lowering makes; the forwarding of materialized tensors: the poly hart graph, its thread subgraphs checked at every
# DDR result; then (configured again once the checks exist) one slice per
# check of each subgraph, cut into batches of one tile iteration, lowered like
# the recipe's harts. The whole graph is lowered to its physical banks, which
# every slice must follow, and the operators of the checked graph must equal
# the compiled model's. Each slice's ELF, with its readback, is poly's (gvn/
# of its device simulation).

# The checks, by the GVN passes; locations kept for matching the GPU's values.
add_custom_command(
  OUTPUT ${DESIGN_OUTPUT}/gvn/laya/npu/checks.mlir
         ${DESIGN_OUTPUT}/gvn/laya/npu/gvn_operators.mlir
         ${DESIGN_OUTPUT}/gvn/laya/npu/model_operators.mlir
  BYPRODUCTS ${DESIGN_OUTPUT}/gvn/laya/npu/hart.mlir
             ${DESIGN_OUTPUT}/gvn/laya/npu/attention_acc.mlir
             ${DESIGN_OUTPUT}/gvn/laya/npu/acc.mlir
             ${DESIGN_OUTPUT}/gvn/laya/npu/frontend.mlir
             ${DESIGN_OUTPUT}/gvn/laya/npu/inlined.mlir
             ${DESIGN_OUTPUT}/gvn/laya/npu/late_attention_acc.mlir
             ${DESIGN_OUTPUT}/gvn/laya/npu/late_acc.mlir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${DESIGN_OUTPUT}/gvn/laya/npu
  COMMAND ${DESIGN_HART_OPT_poly}
            --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so -mlir-print-debuginfo ${OUTPUT_DIR}/poly_linalg.mlir -gvn-source-origins -inline "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${DESIGN_OUTPUT}/gvn/laya/npu/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention}
            --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            -mlir-print-debuginfo ${DESIGN_OUTPUT}/gvn/laya/npu/hart.mlir
            -canonicalize
            -convert-linalg-to-acc
            -o ${DESIGN_OUTPUT}/gvn/laya/npu/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn}
            --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            -mlir-print-debuginfo ${DESIGN_OUTPUT}/gvn/laya/npu/attention_acc.mlir
            -convert-linalg-to-acc
            -o ${DESIGN_OUTPUT}/gvn/laya/npu/acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly}
            --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so -mlir-print-debuginfo ${DESIGN_OUTPUT}/gvn/laya/npu/acc.mlir -lower-frontend-fake-op -o ${DESIGN_OUTPUT}/gvn/laya/npu/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly}
            --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so -mlir-print-debuginfo ${DESIGN_OUTPUT}/gvn/laya/npu/frontend.mlir -inline -eliminate-empty-tensors -o ${DESIGN_OUTPUT}/gvn/laya/npu/inlined.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention}
            --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            -mlir-print-debuginfo ${DESIGN_OUTPUT}/gvn/laya/npu/inlined.mlir "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly"
            -convert-linalg-to-acc
            -o ${DESIGN_OUTPUT}/gvn/laya/npu/late_attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn}
            --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            -mlir-print-debuginfo ${DESIGN_OUTPUT}/gvn/laya/npu/late_attention_acc.mlir
            -convert-linalg-to-acc
            -o ${DESIGN_OUTPUT}/gvn/laya/npu/late_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly}
            --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so -mlir-print-debuginfo ${DESIGN_OUTPUT}/gvn/laya/npu/late_acc.mlir -gvn-insert-checks -o ${DESIGN_OUTPUT}/gvn/laya/npu/checks.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${DESIGN_OUTPUT}/gvn/laya/npu/late_acc.mlir -symbol-privatize=exclude=operator,gp_space -symbol-dce -canonicalize -strip-debuginfo -o ${DESIGN_OUTPUT}/gvn/laya/npu/gvn_operators.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUTPUT_DIR}/operators.mlir -symbol-privatize=exclude=operator,gp_space -symbol-dce -canonicalize -strip-debuginfo -o ${DESIGN_OUTPUT}/gvn/laya/npu/model_operators.mlir
  DEPENDS ${OUTPUT_DIR}/poly_linalg.mlir
          ${OUTPUT_DIR}/operators.mlir
          ${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
          ${DESIGN_HART_PLUGIN_poly}
          ${DESIGN_TARGET_PLUGIN_attention}
          ${DESIGN_TARGET_PLUGIN_ffn}
          ${DESIGN_PB}
  COMMAND_EXPAND_LISTS
  VERBATIM
)
add_custom_target(
  laya-gvn-checks
  DEPENDS
    ${DESIGN_OUTPUT}/gvn/laya/npu/checks.mlir
    ${DESIGN_OUTPUT}/gvn/laya/npu/gvn_operators.mlir
    ${DESIGN_OUTPUT}/gvn/laya/npu/model_operators.mlir
)

# The whole graph, to its physical banks.
add_custom_command(
  OUTPUT ${DESIGN_OUTPUT}/gvn/laya/npu/whole_physical.mlir
  BYPRODUCTS ${DESIGN_OUTPUT}/gvn/laya/npu/whole_frontend.mlir
             ${DESIGN_OUTPUT}/gvn/laya/npu/whole_bufferized.mlir
             ${DESIGN_OUTPUT}/gvn/laya/npu/whole_attention_banks.mlir
             ${DESIGN_OUTPUT}/gvn/laya/npu/whole_attention_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${DESIGN_OUTPUT}/gvn/laya/npu/late_acc.mlir -prune-gp-operators -symbol-dce -inline -eliminate-empty-tensors -forward-materialized-tensors -o ${DESIGN_OUTPUT}/gvn/laya/npu/whole_frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${DESIGN_OUTPUT}/gvn/laya/npu/whole_frontend.mlir -convert-elementwise-to-linalg -resolve-shaped-type-result-dims -canonicalize "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map" -drop-equivalent-buffer-results=modify-public-functions "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs" -ownership-based-buffer-deallocation -buffer-deallocation-simplification -bufferization-lower-deallocations -canonicalize -o ${DESIGN_OUTPUT}/gvn/laya/npu/whole_bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${DESIGN_OUTPUT}/gvn/laya/npu/whole_bufferized.mlir
            -assign-physical-banks
            -o ${DESIGN_OUTPUT}/gvn/laya/npu/whole_attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${DESIGN_OUTPUT}/gvn/laya/npu/whole_attention_banks.mlir
            -lower-acc
            -o ${DESIGN_OUTPUT}/gvn/laya/npu/whole_attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${DESIGN_OUTPUT}/gvn/laya/npu/whole_attention_instructions.mlir
            -assign-physical-banks
            -o ${DESIGN_OUTPUT}/gvn/laya/npu/whole_physical.mlir
  DEPENDS ${DESIGN_OUTPUT}/gvn/laya/npu/late_acc.mlir
          ${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
          ${DESIGN_HART_PLUGIN_poly}
          ${DESIGN_TARGET_PLUGIN_attention}
          ${DESIGN_TARGET_PLUGIN_ffn}
          ${DESIGN_PB}
  COMMAND_EXPAND_LISTS
  VERBATIM
)

# One slice per check of each thread subgraph, once the checks exist: a
# check belongs to the function it is in.
set(objects)
if(EXISTS ${DESIGN_OUTPUT}/gvn/laya/npu/checks.mlir)
  file(STRINGS ${DESIGN_OUTPUT}/gvn/laya/npu/checks.mlir lines REGEX "func\\.func |gvn\\.check ")
  foreach(line ${lines})
    if(line MATCHES "func\\.func (private )?@([A-Za-z0-9_]+)\\(")
      set(function ${CMAKE_MATCH_2})
      set(index 0)
    else()
      set(slice ${DESIGN_OUTPUT}/gvn/laya/${function}.${index})
      add_custom_command(
        OUTPUT ${slice}/slice.o
               ${slice}/slice_physical.mlir
               ${slice}/slice_instructions.mlir
               ${slice}/layout.json
        COMMAND ${CMAKE_COMMAND} -E make_directory ${slice}
        COMMAND ${DESIGN_HART_OPT_poly}
                  --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
                  --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so -mlir-print-debuginfo ${DESIGN_OUTPUT}/gvn/laya/npu/checks.mlir -canonicalize "-gvn-slice=function=${function} index=${index} iterate=true batch=1 rows=32 layout=${slice}/layout.json" -o ${slice}/slice.mlir
        COMMAND ${DESIGN_HART_OPT_poly} ${slice}/slice.mlir -prune-gp-operators -symbol-dce -inline -eliminate-empty-tensors -forward-materialized-tensors -o ${slice}/slice_frontend.mlir
        COMMAND ${DESIGN_HART_OPT_poly} ${slice}/slice_frontend.mlir -convert-elementwise-to-linalg -resolve-shaped-type-result-dims -canonicalize "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map" -drop-equivalent-buffer-results=modify-public-functions "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs" -ownership-based-buffer-deallocation -buffer-deallocation-simplification -bufferization-lower-deallocations -canonicalize -o ${slice}/slice_bufferized.mlir
        COMMAND ${DESIGN_TARGET_OPT_attention} ${slice}/slice_bufferized.mlir
                  -assign-physical-banks
                  -o ${slice}/slice_attention_banks.mlir
        COMMAND ${DESIGN_TARGET_OPT_attention} ${slice}/slice_attention_banks.mlir
                  -lower-acc
                  -o ${slice}/slice_attention_instructions.mlir
        COMMAND ${DESIGN_TARGET_OPT_ffn} ${slice}/slice_attention_instructions.mlir
                  -assign-physical-banks
                  -o ${slice}/slice_physical.mlir
        COMMAND ${DESIGN_TARGET_OPT_ffn} ${slice}/slice_physical.mlir
                  -lower-acc
                  -o ${slice}/slice_instructions.mlir
        COMMAND ${DESIGN_HART_OPT_poly} ${slice}/slice_instructions.mlir -expand-strided-metadata -lower-affine -canonicalize -o ${slice}/slice_addresses.mlir
        COMMAND ${DESIGN_HART_OPT_poly} ${slice}/slice_addresses.mlir "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${slice}/slice_memref.mlir
        COMMAND ${DESIGN_HART_OPT_poly} ${slice}/slice_memref.mlir -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${slice}/slice_copies.mlir
        COMMAND ${DESIGN_HART_OPT_poly} ${slice}/slice_copies.mlir -symbol-privatize=exclude=gvn_entry,gvn_in,gvn_out,gvn_table,_binary_operators_image_start -symbol-dce -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm -reconcile-unrealized-casts -o ${slice}/slice_llvm.mlir
        COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${slice}/slice_llvm.mlir
                  --buddy-to-llvmir -o ${slice}/slice.ll
        COMMAND ${LLVM_OPT} ${slice}/slice.ll -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=default<O2>" -S -o ${slice}/slice_opt.ll
        COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${slice}/slice_opt.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -o ${slice}/slice.o
        # Each intermediate holds the whole module (10-110 MB).
        COMMAND ${CMAKE_COMMAND} -E rm -f ${slice}/slice.mlir ${slice}/slice_frontend.mlir ${slice}/slice_bufferized.mlir ${slice}/slice_attention_banks.mlir ${slice}/slice_attention_instructions.mlir ${slice}/slice_addresses.mlir ${slice}/slice_memref.mlir ${slice}/slice_copies.mlir ${slice}/slice_llvm.mlir ${slice}/slice.ll ${slice}/slice_opt.ll
        DEPENDS ${DESIGN_OUTPUT}/gvn/laya/npu/checks.mlir
                ${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
                ${DESIGN_HART_PLUGIN_poly}
                ${DESIGN_TARGET_PLUGIN_attention}
                ${DESIGN_TARGET_PLUGIN_ffn}
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
  OUTPUT ${DESIGN_OUTPUT}/gvn/laya/npu/operators.embed.o
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/../operators.embed.o ${DESIGN_OUTPUT}/gvn/laya/npu/operators.embed.o
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/../operators.embed.o
  VERBATIM
)
add_custom_target(
  laya-gvn
  DEPENDS ${objects} ${DESIGN_OUTPUT}/gvn/laya/npu/whole_physical.mlir ${DESIGN_OUTPUT}/gvn/laya/npu/operators.embed.o
)
add_dependencies(laya-gvn laya-model-build)
