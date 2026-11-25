# The GVN slices of smolvla-base, in gvn/smolvla-base of the config output: each
# hart graph (vision, step_action, prefix_dense) with its thread subgraphs checked at every
# DDR result; then (configured again once the checks exist) one slice per
# check of each subgraph, cut into batches of one tile iteration, lowered like
# the recipe's graph. The whole graph is lowered to its physical banks, which
# every slice must follow, and the operators of each checked graph must equal
# the compiled model's. Each slice's ELF, with its readback, is piplus's (gvn/
# of its device simulation).

set(objects)

# ===--- vision, on the dense harts ---=== #
# The checks, by the GVN passes; locations kept for matching the GPU's values.
add_custom_command(
  OUTPUT ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_checks.mlir
         ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_gvn_operators.mlir
         ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_model_operators.mlir
  BYPRODUCTS ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_hart.mlir
             ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_acc.mlir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${DESIGN_OUTPUT}/gvn/smolvla-base/npu
  COMMAND ${DESIGN_TARGET_OPT_dense}
            --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            -mlir-print-debuginfo ${OUTPUT_DIR}/vision_linalg.mlir
            -gvn-source-origins "-convert-linalg-to-hart=design=${DESIGN_PB} hart=dense"
            -o ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense}
            --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            -mlir-print-debuginfo ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_hart.mlir
            -convert-linalg-to-acc
            -o ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense}
            --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            -mlir-print-debuginfo ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_acc.mlir
            -gvn-insert-checks
            -o ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_checks.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_acc.mlir
            -symbol-privatize=exclude=operator,gp_space
            -symbol-dce
            -canonicalize
            -strip-debuginfo
            -o ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_gvn_operators.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/vision_operators.mlir
            -symbol-privatize=exclude=operator,gp_space
            -symbol-dce
            -canonicalize
            -strip-debuginfo
            -o ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_model_operators.mlir
  DEPENDS ${OUTPUT_DIR}/vision_linalg.mlir
          ${OUTPUT_DIR}/vision_operators.mlir
          ${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
          ${DESIGN_TARGET_PLUGIN_dense}
          ${DESIGN_PB}
  COMMAND_EXPAND_LISTS
  VERBATIM
)

# The whole graph, to its physical banks.
add_custom_command(
  OUTPUT ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_whole_physical.mlir
  BYPRODUCTS ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_whole_frontend.mlir
             ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_whole_bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_acc.mlir
            -prune-gp-operators
            -symbol-dce
            -lower-frontend-fake-op
            -o ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_whole_frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_whole_frontend.mlir
            -inline
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
            -drop-equivalent-buffer-results=modify-public-functions "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -canonicalize
            -o ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_whole_bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_whole_bufferized.mlir
            -assign-physical-banks
            -o ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_whole_physical.mlir
  DEPENDS ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_acc.mlir
          ${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
          ${DESIGN_TARGET_PLUGIN_dense}
          ${DESIGN_PB}
  COMMAND_EXPAND_LISTS
  VERBATIM
)

# One slice per check of each thread subgraph, once the checks exist: a
# check belongs to the function it is in.
if(EXISTS ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_checks.mlir)
  file(STRINGS ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_checks.mlir lines REGEX "func\\.func |gvn\\.check ")
  foreach(line ${lines})
    if(line MATCHES "func\\.func (private )?@([A-Za-z0-9_]+)\\(")
      set(function ${CMAKE_MATCH_2})
      set(index 0)
    else()
      set(slice ${DESIGN_OUTPUT}/gvn/smolvla-base/${function}.${index})
      add_custom_command(
        OUTPUT ${slice}/slice.o
               ${slice}/slice_physical.mlir
               ${slice}/layout.json
        COMMAND ${CMAKE_COMMAND} -E make_directory ${slice}
        COMMAND ${DESIGN_TARGET_OPT_dense}
                  --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
                  --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
                  -mlir-print-debuginfo ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_checks.mlir
                  -canonicalize "-gvn-slice=function=${function} index=${index} iterate=true batch=1 layout=${slice}/layout.json"
                  -o ${slice}/slice.mlir
        COMMAND ${DESIGN_TARGET_OPT_dense} ${slice}/slice.mlir
                  -prune-gp-operators
                  -symbol-dce
                  -lower-frontend-fake-op
                  -o ${slice}/slice_frontend.mlir
        COMMAND ${DESIGN_TARGET_OPT_dense} ${slice}/slice_frontend.mlir
                  -inline
                  -eliminate-empty-tensors
                  -convert-elementwise-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
                  -drop-equivalent-buffer-results=modify-public-functions "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
                  -ownership-based-buffer-deallocation
                  -buffer-deallocation-simplification
                  -bufferization-lower-deallocations
                  -canonicalize
                  -o ${slice}/slice_bufferized.mlir
        COMMAND ${DESIGN_TARGET_OPT_dense} ${slice}/slice_bufferized.mlir
                  -assign-physical-banks
                  -o ${slice}/slice_physical.mlir
        COMMAND ${DESIGN_TARGET_OPT_dense} ${slice}/slice_physical.mlir
                  -lower-acc
                  -o ${slice}/slice_instructions.mlir
        COMMAND ${DESIGN_TARGET_OPT_dense} ${slice}/slice_instructions.mlir
                  -expand-strided-metadata
                  -lower-affine
                  -canonicalize
                  -o ${slice}/slice_addresses.mlir
        COMMAND ${DESIGN_TARGET_OPT_dense} ${slice}/slice_addresses.mlir "-lower-hart=design=${DESIGN_PB} hart=dense"
                  -o ${slice}/slice_memref.mlir
        COMMAND ${DESIGN_TARGET_OPT_dense} ${slice}/slice_memref.mlir
                  -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))"
                  -o ${slice}/slice_copies.mlir
        COMMAND ${DESIGN_TARGET_OPT_dense} ${slice}/slice_copies.mlir
                  -symbol-privatize=exclude=gvn_entry,gvn_in,gvn_out,gvn_table,_binary_operators_image_start
                  -symbol-dce
                  -convert-linalg-to-loops
                  -expand-strided-metadata
                  -lower-affine
                  -convert-scf-to-cf
                  -finalize-memref-to-llvm
                  -convert-cf-to-llvm
                  -convert-arith-to-llvm
                  -convert-math-to-llvm
                  -convert-math-to-libm
                  -convert-func-to-llvm
                  -reconcile-unrealized-casts
                  -o ${slice}/slice_llvm.mlir
        COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${slice}/slice_llvm.mlir
                  --buddy-to-llvmir -o ${slice}/slice.ll
        COMMAND ${LLVM_OPT} ${slice}/slice.ll -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=default<O2>" -S -o ${slice}/slice_opt.ll
        COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${slice}/slice_opt.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -max-store-memcpy=16 -o ${slice}/slice.o
        COMMAND ${ELF_OBJCOPY}
                  --redefine-sym _binary_operators_image_start=_binary_vision_operators_image_start ${slice}/slice.o
        # Each intermediate holds the whole module.
        COMMAND ${CMAKE_COMMAND} -E rm -f ${slice}/slice.mlir ${slice}/slice_frontend.mlir ${slice}/slice_bufferized.mlir ${slice}/slice_instructions.mlir ${slice}/slice_addresses.mlir ${slice}/slice_memref.mlir ${slice}/slice_copies.mlir ${slice}/slice_llvm.mlir ${slice}/slice.ll ${slice}/slice_opt.ll
        DEPENDS ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_checks.mlir
                ${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
                ${DESIGN_TARGET_PLUGIN_dense}
                ${DESIGN_PB}
        COMMAND_EXPAND_LISTS
        VERBATIM
      )
      list(APPEND objects ${slice}/slice.o)
      math(EXPR index "${index} + 1")
    endif()
  endforeach()
endif()

# The graph's operators, for the slice ELFs.
add_custom_command(
  OUTPUT ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_operators.o
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/../vision_operators.o ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_operators.o
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/../vision_operators.o
  VERBATIM
)

# ===--- step_action, on the action harts ---=== #
# The checks, by the GVN passes; locations kept for matching the GPU's values.
add_custom_command(
  OUTPUT ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_checks.mlir
         ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_gvn_operators.mlir
         ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_model_operators.mlir
  BYPRODUCTS ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_hart.mlir
             ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_acc.mlir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${DESIGN_OUTPUT}/gvn/smolvla-base/npu
  COMMAND ${DESIGN_TARGET_OPT_action}
            --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            -mlir-print-debuginfo ${OUTPUT_DIR}/step_action_linalg.mlir
            -gvn-source-origins "-convert-linalg-to-hart=design=${DESIGN_PB} hart=action"
            -o ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_action}
            --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            -mlir-print-debuginfo ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_hart.mlir
            -convert-linalg-to-acc
            -o ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_action}
            --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            -mlir-print-debuginfo ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_acc.mlir
            -gvn-insert-checks
            -o ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_checks.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_acc.mlir
            -symbol-privatize=exclude=operator,gp_space
            -symbol-dce
            -canonicalize
            -strip-debuginfo
            -o ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_gvn_operators.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/step_action_operators.mlir
            -symbol-privatize=exclude=operator,gp_space
            -symbol-dce
            -canonicalize
            -strip-debuginfo
            -o ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_model_operators.mlir
  DEPENDS ${OUTPUT_DIR}/step_action_linalg.mlir
          ${OUTPUT_DIR}/step_action_operators.mlir
          ${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
          ${DESIGN_TARGET_PLUGIN_action}
          ${DESIGN_PB}
  COMMAND_EXPAND_LISTS
  VERBATIM
)

# The whole graph, to its physical banks.
add_custom_command(
  OUTPUT ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_whole_physical.mlir
  BYPRODUCTS ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_whole_frontend.mlir
             ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_whole_bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_acc.mlir
            -prune-gp-operators
            -symbol-dce
            -lower-frontend-fake-op
            -o ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_whole_frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_whole_frontend.mlir
            -inline
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
            -drop-equivalent-buffer-results=modify-public-functions "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -canonicalize
            -o ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_whole_bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_whole_bufferized.mlir
            -assign-physical-banks
            -o ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_whole_physical.mlir
  DEPENDS ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_acc.mlir
          ${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
          ${DESIGN_TARGET_PLUGIN_action}
          ${DESIGN_PB}
  COMMAND_EXPAND_LISTS
  VERBATIM
)

# One slice per check of each thread subgraph, once the checks exist: a
# check belongs to the function it is in.
if(EXISTS ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_checks.mlir)
  file(STRINGS ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_checks.mlir lines REGEX "func\\.func |gvn\\.check ")
  foreach(line ${lines})
    if(line MATCHES "func\\.func (private )?@([A-Za-z0-9_]+)\\(")
      set(function ${CMAKE_MATCH_2})
      set(index 0)
    else()
      set(slice ${DESIGN_OUTPUT}/gvn/smolvla-base/${function}.${index})
      add_custom_command(
        OUTPUT ${slice}/slice.o
               ${slice}/slice_physical.mlir
               ${slice}/layout.json
        COMMAND ${CMAKE_COMMAND} -E make_directory ${slice}
        COMMAND ${DESIGN_TARGET_OPT_action}
                  --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
                  --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
                  -mlir-print-debuginfo ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_checks.mlir
                  -canonicalize "-gvn-slice=function=${function} index=${index} iterate=true batch=1 layout=${slice}/layout.json"
                  -o ${slice}/slice.mlir
        COMMAND ${DESIGN_TARGET_OPT_action} ${slice}/slice.mlir
                  -prune-gp-operators
                  -symbol-dce
                  -lower-frontend-fake-op
                  -o ${slice}/slice_frontend.mlir
        COMMAND ${DESIGN_TARGET_OPT_action} ${slice}/slice_frontend.mlir
                  -inline
                  -eliminate-empty-tensors
                  -convert-elementwise-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
                  -drop-equivalent-buffer-results=modify-public-functions "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
                  -ownership-based-buffer-deallocation
                  -buffer-deallocation-simplification
                  -bufferization-lower-deallocations
                  -canonicalize
                  -o ${slice}/slice_bufferized.mlir
        COMMAND ${DESIGN_TARGET_OPT_action} ${slice}/slice_bufferized.mlir
                  -assign-physical-banks
                  -o ${slice}/slice_physical.mlir
        COMMAND ${DESIGN_TARGET_OPT_action} ${slice}/slice_physical.mlir
                  -lower-acc
                  -o ${slice}/slice_instructions.mlir
        COMMAND ${DESIGN_TARGET_OPT_action} ${slice}/slice_instructions.mlir
                  -expand-strided-metadata
                  -lower-affine
                  -canonicalize
                  -o ${slice}/slice_addresses.mlir
        COMMAND ${DESIGN_TARGET_OPT_action} ${slice}/slice_addresses.mlir "-lower-hart=design=${DESIGN_PB} hart=action"
                  -o ${slice}/slice_memref.mlir
        COMMAND ${DESIGN_TARGET_OPT_action} ${slice}/slice_memref.mlir
                  -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))"
                  -o ${slice}/slice_copies.mlir
        COMMAND ${DESIGN_TARGET_OPT_action} ${slice}/slice_copies.mlir
                  -symbol-privatize=exclude=gvn_entry,gvn_in,gvn_out,gvn_table,_binary_operators_image_start
                  -symbol-dce
                  -convert-linalg-to-loops
                  -expand-strided-metadata
                  -lower-affine
                  -convert-scf-to-cf
                  -finalize-memref-to-llvm
                  -convert-cf-to-llvm
                  -convert-arith-to-llvm
                  -convert-math-to-llvm
                  -convert-math-to-libm
                  -convert-func-to-llvm
                  -reconcile-unrealized-casts
                  -o ${slice}/slice_llvm.mlir
        COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${slice}/slice_llvm.mlir
                  --buddy-to-llvmir -o ${slice}/slice.ll
        COMMAND ${LLVM_OPT} ${slice}/slice.ll -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=default<O2>" -S -o ${slice}/slice_opt.ll
        COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${slice}/slice_opt.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -max-store-memcpy=16 -o ${slice}/slice.o
        COMMAND ${ELF_OBJCOPY}
                  --redefine-sym _binary_operators_image_start=_binary_step_action_operators_image_start ${slice}/slice.o
        # Each intermediate holds the whole module.
        COMMAND ${CMAKE_COMMAND} -E rm -f ${slice}/slice.mlir ${slice}/slice_frontend.mlir ${slice}/slice_bufferized.mlir ${slice}/slice_instructions.mlir ${slice}/slice_addresses.mlir ${slice}/slice_memref.mlir ${slice}/slice_copies.mlir ${slice}/slice_llvm.mlir ${slice}/slice.ll ${slice}/slice_opt.ll
        DEPENDS ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_checks.mlir
                ${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
                ${DESIGN_TARGET_PLUGIN_action}
                ${DESIGN_PB}
        COMMAND_EXPAND_LISTS
        VERBATIM
      )
      list(APPEND objects ${slice}/slice.o)
      math(EXPR index "${index} + 1")
    endif()
  endforeach()
endif()

# The graph's operators, for the slice ELFs.
add_custom_command(
  OUTPUT ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_operators.o
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/../step_action_operators.o ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_operators.o
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/../step_action_operators.o
  VERBATIM
)

# ===--- prefix_dense, on the dense harts ---=== #
# The checks, by the GVN passes; locations kept for matching the GPU's values.
add_custom_command(
  OUTPUT ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_checks.mlir
         ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_gvn_operators.mlir
         ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_model_operators.mlir
  BYPRODUCTS ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_hart.mlir
             ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_acc.mlir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${DESIGN_OUTPUT}/gvn/smolvla-base/npu
  COMMAND ${DESIGN_TARGET_OPT_dense}
            --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            -mlir-print-debuginfo ${OUTPUT_DIR}/prefix_dense_linalg.mlir
            -gvn-source-origins "-convert-linalg-to-hart=design=${DESIGN_PB} hart=dense"
            -o ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense}
            --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            -mlir-print-debuginfo ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_hart.mlir
            -convert-linalg-to-acc
            -o ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense}
            --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            -mlir-print-debuginfo ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_acc.mlir
            -gvn-insert-checks
            -o ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_checks.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_acc.mlir
            -symbol-privatize=exclude=operator,gp_space
            -symbol-dce
            -canonicalize
            -strip-debuginfo
            -o ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_gvn_operators.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/prefix_dense_operators.mlir
            -symbol-privatize=exclude=operator,gp_space
            -symbol-dce
            -canonicalize
            -strip-debuginfo
            -o ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_model_operators.mlir
  DEPENDS ${OUTPUT_DIR}/prefix_dense_linalg.mlir
          ${OUTPUT_DIR}/prefix_dense_operators.mlir
          ${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
          ${DESIGN_TARGET_PLUGIN_dense}
          ${DESIGN_PB}
  COMMAND_EXPAND_LISTS
  VERBATIM
)

# The whole graph, to its physical banks.
add_custom_command(
  OUTPUT ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_whole_physical.mlir
  BYPRODUCTS ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_whole_frontend.mlir
             ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_whole_bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_acc.mlir
            -prune-gp-operators
            -symbol-dce
            -lower-frontend-fake-op
            -o ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_whole_frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_whole_frontend.mlir
            -inline
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
            -drop-equivalent-buffer-results=modify-public-functions "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -canonicalize
            -o ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_whole_bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_whole_bufferized.mlir
            -assign-physical-banks
            -o ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_whole_physical.mlir
  DEPENDS ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_acc.mlir
          ${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
          ${DESIGN_TARGET_PLUGIN_dense}
          ${DESIGN_PB}
  COMMAND_EXPAND_LISTS
  VERBATIM
)

# One slice per check of each thread subgraph, once the checks exist: a
# check belongs to the function it is in.
if(EXISTS ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_checks.mlir)
  file(STRINGS ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_checks.mlir lines REGEX "func\\.func |gvn\\.check ")
  foreach(line ${lines})
    if(line MATCHES "func\\.func (private )?@([A-Za-z0-9_]+)\\(")
      set(function ${CMAKE_MATCH_2})
      set(index 0)
    else()
      set(slice ${DESIGN_OUTPUT}/gvn/smolvla-base/${function}.${index})
      add_custom_command(
        OUTPUT ${slice}/slice.o
               ${slice}/slice_physical.mlir
               ${slice}/layout.json
        COMMAND ${CMAKE_COMMAND} -E make_directory ${slice}
        COMMAND ${DESIGN_TARGET_OPT_dense}
                  --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
                  --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
                  -mlir-print-debuginfo ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_checks.mlir
                  -canonicalize "-gvn-slice=function=${function} index=${index} iterate=true batch=1 layout=${slice}/layout.json"
                  -o ${slice}/slice.mlir
        COMMAND ${DESIGN_TARGET_OPT_dense} ${slice}/slice.mlir
                  -prune-gp-operators
                  -symbol-dce
                  -lower-frontend-fake-op
                  -o ${slice}/slice_frontend.mlir
        COMMAND ${DESIGN_TARGET_OPT_dense} ${slice}/slice_frontend.mlir
                  -inline
                  -eliminate-empty-tensors
                  -convert-elementwise-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
                  -drop-equivalent-buffer-results=modify-public-functions "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
                  -ownership-based-buffer-deallocation
                  -buffer-deallocation-simplification
                  -bufferization-lower-deallocations
                  -canonicalize
                  -o ${slice}/slice_bufferized.mlir
        COMMAND ${DESIGN_TARGET_OPT_dense} ${slice}/slice_bufferized.mlir
                  -assign-physical-banks
                  -o ${slice}/slice_physical.mlir
        COMMAND ${DESIGN_TARGET_OPT_dense} ${slice}/slice_physical.mlir
                  -lower-acc
                  -o ${slice}/slice_instructions.mlir
        COMMAND ${DESIGN_TARGET_OPT_dense} ${slice}/slice_instructions.mlir
                  -expand-strided-metadata
                  -lower-affine
                  -canonicalize
                  -o ${slice}/slice_addresses.mlir
        COMMAND ${DESIGN_TARGET_OPT_dense} ${slice}/slice_addresses.mlir "-lower-hart=design=${DESIGN_PB} hart=dense"
                  -o ${slice}/slice_memref.mlir
        COMMAND ${DESIGN_TARGET_OPT_dense} ${slice}/slice_memref.mlir
                  -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))"
                  -o ${slice}/slice_copies.mlir
        COMMAND ${DESIGN_TARGET_OPT_dense} ${slice}/slice_copies.mlir
                  -symbol-privatize=exclude=gvn_entry,gvn_in,gvn_out,gvn_table,_binary_operators_image_start
                  -symbol-dce
                  -convert-linalg-to-loops
                  -expand-strided-metadata
                  -lower-affine
                  -convert-scf-to-cf
                  -finalize-memref-to-llvm
                  -convert-cf-to-llvm
                  -convert-arith-to-llvm
                  -convert-math-to-llvm
                  -convert-math-to-libm
                  -convert-func-to-llvm
                  -reconcile-unrealized-casts
                  -o ${slice}/slice_llvm.mlir
        COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${slice}/slice_llvm.mlir
                  --buddy-to-llvmir -o ${slice}/slice.ll
        COMMAND ${LLVM_OPT} ${slice}/slice.ll -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=default<O2>" -S -o ${slice}/slice_opt.ll
        COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${slice}/slice_opt.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -max-store-memcpy=16 -o ${slice}/slice.o
        COMMAND ${ELF_OBJCOPY}
                  --redefine-sym _binary_operators_image_start=_binary_prefix_dense_operators_image_start ${slice}/slice.o
        # Each intermediate holds the whole module.
        COMMAND ${CMAKE_COMMAND} -E rm -f ${slice}/slice.mlir ${slice}/slice_frontend.mlir ${slice}/slice_bufferized.mlir ${slice}/slice_instructions.mlir ${slice}/slice_addresses.mlir ${slice}/slice_memref.mlir ${slice}/slice_copies.mlir ${slice}/slice_llvm.mlir ${slice}/slice.ll ${slice}/slice_opt.ll
        DEPENDS ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_checks.mlir
                ${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
                ${DESIGN_TARGET_PLUGIN_dense}
                ${DESIGN_PB}
        COMMAND_EXPAND_LISTS
        VERBATIM
      )
      list(APPEND objects ${slice}/slice.o)
      math(EXPR index "${index} + 1")
    endif()
  endforeach()
endif()

# The graph's operators, for the slice ELFs.
add_custom_command(
  OUTPUT ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_operators.o
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/../prefix_dense_operators.o ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_operators.o
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/../prefix_dense_operators.o
  VERBATIM
)

add_custom_target(
  smolvla-base-gvn-checks
  DEPENDS
    ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_checks.mlir
    ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_gvn_operators.mlir
    ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_model_operators.mlir
    ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_checks.mlir
    ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_gvn_operators.mlir
    ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_model_operators.mlir
    ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_checks.mlir
    ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_gvn_operators.mlir
    ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_model_operators.mlir
)
add_custom_target(
  smolvla-base-gvn
  DEPENDS
    ${objects}
    ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_whole_physical.mlir
    ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/vision_operators.o
    ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_whole_physical.mlir
    ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/step_action_operators.o
    ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_whole_physical.mlir
    ${DESIGN_OUTPUT}/gvn/smolvla-base/npu/prefix_dense_operators.o
)
add_dependencies(smolvla-base-gvn smolvla-base-model-build)
