# The GVN slices of lenet, in gvn/lenet of the config output: the hart graph
# with a check at every DDR result, then (configured again once the checks
# exist) one slice per check, cut into batches of loop iterations, each
# lowered like the recipe's subgraph and linked with the entry, so LLVM keeps
# only what the slice uses; the whole graph is lowered to its physical banks,
# which every slice must match. Then one ELF per slice, with its GPU values.

# The checks, by the GVN passes; locations kept for matching the GPU's values.
add_custom_command(
  OUTPUT ${DESIGN_OUTPUT}/gvn/lenet/npu/checks.mlir
  BYPRODUCTS ${DESIGN_OUTPUT}/gvn/lenet/npu/acc.mlir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${DESIGN_OUTPUT}/gvn/lenet/npu
  COMMAND ${DESIGN_TARGET_OPT_toy}
            --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            -mlir-print-debuginfo ${OUTPUT_DIR}/subgraph0_linalg.mlir
            -convert-linalg-to-acc
            -materialize-acc-results
            -o ${DESIGN_OUTPUT}/gvn/lenet/npu/acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_toy}
            --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            -mlir-print-debuginfo ${DESIGN_OUTPUT}/gvn/lenet/npu/acc.mlir "-gvn-insert-op-checks=ops=bufferization.to_tensor"
            -gvn-insert-checks
            -o ${DESIGN_OUTPUT}/gvn/lenet/npu/checks.mlir
  DEPENDS ${OUTPUT_DIR}/subgraph0_linalg.mlir
          ${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
          ${DESIGN_TARGET_PLUGIN_toy}
          ${DESIGN_PB}
  COMMAND_EXPAND_LISTS
  VERBATIM
)
add_custom_target(lenet-gvn-checks DEPENDS ${DESIGN_OUTPUT}/gvn/lenet/npu/checks.mlir)

# The entry, as the recipe compiles it.
add_custom_command(
  OUTPUT ${DESIGN_OUTPUT}/gvn/lenet/npu/forward.ll
  BYPRODUCTS ${DESIGN_OUTPUT}/gvn/lenet/npu/forward_llvm.mlir
  COMMAND ${DESIGN_TARGET_OPT_toy} ${OUTPUT_DIR}/forward.mlir
            -buffer-deallocation-simplification
            -convert-linalg-to-loops
            -eliminate-empty-tensors
            -llvm-request-c-wrappers
            -convert-math-to-llvm
            -convert-math-to-libm
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-arith-to-llvm
            -expand-strided-metadata
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts
            -o ${DESIGN_OUTPUT}/gvn/lenet/npu/forward_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${DESIGN_OUTPUT}/gvn/lenet/npu/forward_llvm.mlir
            --buddy-to-llvmir -o ${DESIGN_OUTPUT}/gvn/lenet/npu/forward.ll
  DEPENDS ${OUTPUT_DIR}/forward.mlir
          ${DESIGN_TARGET_PLUGIN_toy}
          ${DESIGN_PB}
  COMMAND_EXPAND_LISTS
  VERBATIM
)

# The whole graph, to its physical banks.
add_custom_command(
  OUTPUT ${DESIGN_OUTPUT}/gvn/lenet/npu/whole_physical.mlir
  BYPRODUCTS ${DESIGN_OUTPUT}/gvn/lenet/npu/whole_frontend.mlir
             ${DESIGN_OUTPUT}/gvn/lenet/npu/whole_bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_toy} ${DESIGN_OUTPUT}/gvn/lenet/npu/acc.mlir
            -lower-frontend-fake-op
            -o ${DESIGN_OUTPUT}/gvn/lenet/npu/whole_frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_toy} ${DESIGN_OUTPUT}/gvn/lenet/npu/whole_frontend.mlir
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -linalg-fuse-elementwise-ops
            -canonicalize
            -cse
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-bufferization-to-memref
            -o ${DESIGN_OUTPUT}/gvn/lenet/npu/whole_bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_toy} ${DESIGN_OUTPUT}/gvn/lenet/npu/whole_bufferized.mlir
            -cse
            -canonicalize
            -assign-physical-banks
            -o ${DESIGN_OUTPUT}/gvn/lenet/npu/whole_physical.mlir
  DEPENDS ${DESIGN_OUTPUT}/gvn/lenet/npu/acc.mlir
          ${DESIGN_TARGET_PLUGIN_toy}
          ${DESIGN_PB}
  COMMAND_EXPAND_LISTS
  VERBATIM
)

# One slice per check, once the checks exist.
set(slices)
if(EXISTS ${DESIGN_OUTPUT}/gvn/lenet/npu/checks.mlir)
  file(STRINGS ${DESIGN_OUTPUT}/gvn/lenet/npu/checks.mlir checks REGEX "gvn\\.check ")
  list(LENGTH checks count)
  math(EXPR last "${count} - 1")
  foreach(index RANGE ${last})
    list(APPEND slices ${DESIGN_OUTPUT}/gvn/lenet/slice${index})
    add_custom_command(
      OUTPUT ${DESIGN_OUTPUT}/gvn/lenet/slice${index}/slice.mlir
             ${DESIGN_OUTPUT}/gvn/lenet/slice${index}/layout.json
      COMMAND ${CMAKE_COMMAND} -E make_directory ${DESIGN_OUTPUT}/gvn/lenet/slice${index}
      COMMAND ${DESIGN_TARGET_OPT_toy}
                --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
                --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
                -mlir-print-debuginfo ${DESIGN_OUTPUT}/gvn/lenet/npu/checks.mlir "-gvn-slice=function=subgraph0 index=${index} iterate=true batch=1 layout=${DESIGN_OUTPUT}/gvn/lenet/slice${index}/layout.json"
                -o ${DESIGN_OUTPUT}/gvn/lenet/slice${index}/slice.mlir
      DEPENDS ${DESIGN_OUTPUT}/gvn/lenet/npu/checks.mlir
              ${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
              ${DESIGN_TARGET_PLUGIN_toy}
              ${DESIGN_PB}
      COMMAND_EXPAND_LISTS
      VERBATIM
    )
  endforeach()
endif()
set(objects)
foreach(slice ${slices})
  add_custom_command(
    OUTPUT ${slice}/slice.o
           ${slice}/slice_physical.mlir
    BYPRODUCTS ${slice}/slice_frontend.mlir
               ${slice}/slice_bufferized.mlir
               ${slice}/slice_cwrappers.mlir
               ${slice}/slice_lowered.mlir
               ${slice}/slice_llvm.mlir
               ${slice}/slice.ll
               ${slice}/slice_linked.ll
               ${slice}/slice_opt.ll
    COMMAND ${DESIGN_TARGET_OPT_toy} ${slice}/slice.mlir
              -lower-frontend-fake-op
              -o ${slice}/slice_frontend.mlir
    COMMAND ${DESIGN_TARGET_OPT_toy} ${slice}/slice_frontend.mlir
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -linalg-fuse-elementwise-ops
              -canonicalize
              -cse
              -one-shot-bufferize=bufferize-function-boundaries
              -ownership-based-buffer-deallocation
              -buffer-deallocation-simplification
              -bufferization-lower-deallocations
              -convert-bufferization-to-memref
              -o ${slice}/slice_bufferized.mlir
    COMMAND ${DESIGN_TARGET_OPT_toy} ${slice}/slice_bufferized.mlir
              -cse
              -canonicalize
              -assign-physical-banks
              -o ${slice}/slice_physical.mlir
    COMMAND ${DESIGN_TARGET_OPT_toy} ${slice}/slice_physical.mlir
              -llvm-request-c-wrappers
              -o ${slice}/slice_cwrappers.mlir
    COMMAND ${DESIGN_TARGET_OPT_toy} ${slice}/slice_cwrappers.mlir
              -lower-acc
              -o ${slice}/slice_lowered.mlir
    COMMAND ${DESIGN_TARGET_OPT_toy} ${slice}/slice_lowered.mlir
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
              -o ${slice}/slice_llvm.mlir
    COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${slice}/slice_llvm.mlir
              --buddy-to-llvmir -o ${slice}/slice.ll
    # Let LLVM eliminate the entry's unused tensor descriptors for this slice.
    COMMAND ${BUDDY_SOURCE}/llvm/build/bin/llvm-link ${slice}/slice.ll ${DESIGN_OUTPUT}/gvn/lenet/npu/forward.ll -S -o ${slice}/slice_linked.ll
    COMMAND ${LLVM_OPT} ${slice}/slice_linked.ll -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=internalize,default<O2>" -internalize-public-api-list=_mlir_ciface_forward -S -o ${slice}/slice_opt.ll
    COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${slice}/slice_opt.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o ${slice}/slice.o
    DEPENDS ${slice}/slice.mlir
            ${DESIGN_OUTPUT}/gvn/lenet/npu/forward.ll
            ${DESIGN_TARGET_PLUGIN_toy}
            ${DESIGN_PB}
    COMMAND_EXPAND_LISTS
    VERBATIM
  )
  list(APPEND objects ${slice}/slice.o)
endforeach()

add_custom_target(lenet-gvn DEPENDS ${objects} ${DESIGN_OUTPUT}/gvn/lenet/npu/whole_physical.mlir)

# One ELF per slice: the harness with the slice's GPU values embedded
# (slice<i>/gvn_*.bin, written by the driver once the GPU has run), linked
# like the recipe's baremetal ELF.
add_custom_command(
  OUTPUT ${DESIGN_OUTPUT}/gvn/lenet/npu/mlir-baremetal.o
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++17 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -fno-builtin -I${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine -c ${RUNTIME_DIR}/services/mlir.cpp -o ${DESIGN_OUTPUT}/gvn/lenet/npu/mlir-baremetal.o
  DEPENDS ${RUNTIME_DIR}/services/mlir.cpp
  VERBATIM
)
set(elfs)
foreach(slice ${slices})
  add_custom_command(
    OUTPUT ${slice}/slice.elf
    BYPRODUCTS ${slice}/npu.o
    COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++20 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -fno-builtin ${DESIGN_DEFINES} -I${LIB_DIR}
              --embed-dir=${slice}
              --embed-dir=${OUTPUT_DIR} -c ${CMAKE_CURRENT_SOURCE_DIR}/npu.cpp -o ${slice}/npu.o
    COMMAND ${ELF_CC} -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -static -specs=nano.specs -specs=nosys.specs -nostartfiles -Wl,-T,${BOOT_DIR}/device.ld ${DESIGN_DEFINES} ${TARGET_DEFINES} -I${RUNTIME_DIR}/services -I${STACK_ROOT}/kernel/include ${BOOT_DIR}/start.S ${BOOT_DIR}/syscalls.c ${STACK_ROOT}/kernel/src/location.c ${RUNTIME_DIR}/trace/baremetal.c ${slice}/npu.o ${DESIGN_OUTPUT}/gvn/lenet/npu/mlir-baremetal.o ${slice}/slice.o -lm -lgcc -o ${slice}/slice.elf
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/npu.cpp
            ${slice}/gvn_in.bin
            ${slice}/gvn_table.bin
            ${slice}/gvn_expect.bin
            ${slice}/gvn_initial.bin
            ${slice}/slice.o
            ${DESIGN_OUTPUT}/gvn/lenet/npu/mlir-baremetal.o
            ${BOOT_DIR}/start.S
            ${BOOT_DIR}/syscalls.c
            ${BOOT_DIR}/device.ld
            ${STACK_ROOT}/kernel/src/location.c
            ${RUNTIME_DIR}/trace/baremetal.c
    COMMAND_EXPAND_LISTS
    VERBATIM
  )
  list(APPEND elfs ${slice}/slice.elf)
endforeach()
add_custom_target(lenet-gvn-elf DEPENDS ${elfs})
