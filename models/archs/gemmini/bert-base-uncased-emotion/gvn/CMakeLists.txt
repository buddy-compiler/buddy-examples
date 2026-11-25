# The GVN slices of bert-base-uncased-emotion, in gvn/bert-base-uncased-emotion of the config output: the linalg
# graph with a check at every operation Gemmini takes, then (configured
# again once the checks exist) one slice per check, that operation alone (or
# its first tile, for a long one) on its GPU operands, lowered like the
# recipe's subgraph. Then one ELF per
# slice, with its GPU values.

# The checks; locations kept for matching the GPU's values.
add_custom_command(
  OUTPUT ${DESIGN_OUTPUT}/gvn/bert-base-uncased-emotion/npu/checks.mlir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${DESIGN_OUTPUT}/gvn/bert-base-uncased-emotion/npu
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt
            --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
            -mlir-print-debuginfo ${OUTPUT_DIR}/subgraph0_linalg.mlir "-gvn-insert-op-checks=ops=linalg.matmul,linalg.batch_matmul,linalg.batch_matmul_transpose_b,linalg.conv_2d_nhwc_fhwc,linalg.conv_2d_nchw_fchw,linalg.conv_2d_nhwc_hwcf"
            -o ${DESIGN_OUTPUT}/gvn/bert-base-uncased-emotion/npu/checks.mlir
  DEPENDS ${OUTPUT_DIR}/subgraph0_linalg.mlir
          ${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
  VERBATIM
)
add_custom_target(
  bert-base-uncased-emotion-gvn-checks
  DEPENDS ${DESIGN_OUTPUT}/gvn/bert-base-uncased-emotion/npu/checks.mlir
)

# One slice per check, once the driver prepared them (its slice<i>/parts):
# a model build configures without them.
set(slices)
if(EXISTS ${DESIGN_OUTPUT}/gvn/bert-base-uncased-emotion/slice0/parts)
  file(STRINGS ${DESIGN_OUTPUT}/gvn/bert-base-uncased-emotion/npu/checks.mlir checks REGEX "gvn\\.check ")
  list(LENGTH checks count)
  math(EXPR last "${count} - 1")
  foreach(index RANGE ${last})
    list(APPEND slices ${DESIGN_OUTPUT}/gvn/bert-base-uncased-emotion/slice${index})
    # The tiles of its operation, which the driver writes (1: the whole).
    file(READ ${DESIGN_OUTPUT}/gvn/bert-base-uncased-emotion/slice${index}/parts parts)
    string(STRIP "${parts}" parts)
    add_custom_command(
      OUTPUT ${DESIGN_OUTPUT}/gvn/bert-base-uncased-emotion/slice${index}/slice.mlir
             ${DESIGN_OUTPUT}/gvn/bert-base-uncased-emotion/slice${index}/layout.json
      COMMAND ${CMAKE_COMMAND} -E make_directory ${DESIGN_OUTPUT}/gvn/bert-base-uncased-emotion/slice${index}
      COMMAND ${COMPILER_BUILD}/bin/buddy-opt
                --load-dialect-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
                --load-pass-plugin=${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so ${DESIGN_OUTPUT}/gvn/bert-base-uncased-emotion/npu/checks.mlir "-gvn-slice-op=index=${index} parts=${parts} layout=${DESIGN_OUTPUT}/gvn/bert-base-uncased-emotion/slice${index}/layout.json"
                -o ${DESIGN_OUTPUT}/gvn/bert-base-uncased-emotion/slice${index}/slice.mlir
      DEPENDS ${DESIGN_OUTPUT}/gvn/bert-base-uncased-emotion/npu/checks.mlir
              ${DESIGN_OUTPUT}/gvn/bert-base-uncased-emotion/slice${index}/parts
              ${STACK_ROOT}/gvn/build/gvn-compiler/libgvn-compiler.so
      VERBATIM
    )
  endforeach()
endif()
set(objects)
foreach(slice ${slices})
  add_custom_command(
    OUTPUT ${slice}/slice.o
    BYPRODUCTS ${slice}/slice_bufferized.mlir
               ${slice}/slice_gemmini.mlir
               ${slice}/slice_cwrappers.mlir
               ${slice}/slice_lowered.mlir
               ${slice}/slice_llvm.mlir
               ${slice}/slice.ll
    COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${slice}/slice.mlir
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -one-shot-bufferize=bufferize-function-boundaries
              -ownership-based-buffer-deallocation
              -buffer-deallocation-simplification
              -bufferization-lower-deallocations
              -o ${slice}/slice_bufferized.mlir
    COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${slice}/slice_bufferized.mlir
              -convert-linalg-to-gemmini
              -o ${slice}/slice_gemmini.mlir
    COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${slice}/slice_gemmini.mlir
              -expand-strided-metadata
              -convert-linalg-to-loops
              -convert-scf-to-cf
              -llvm-request-c-wrappers
              -o ${slice}/slice_cwrappers.mlir
    COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${slice}/slice_cwrappers.mlir
              -lower-gemmini
              -o ${slice}/slice_lowered.mlir
    COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${slice}/slice_lowered.mlir
              -convert-math-to-llvm
              -convert-math-to-libm
              -convert-arith-to-llvm
              -convert-func-to-llvm
              -finalize-memref-to-llvm
              -reconcile-unrealized-casts
              -o ${slice}/slice_llvm.mlir
    COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${slice}/slice_llvm.mlir
              --buddy-to-llvmir -o ${slice}/slice.ll
    COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${slice}/slice.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -O2 -mattr=+m,+d,+xgemmini -float-abi=hard -o ${slice}/slice.o
    DEPENDS ${slice}/slice.mlir
    VERBATIM
  )
  list(APPEND objects ${slice}/slice.o)
endforeach()
add_custom_target(bert-base-uncased-emotion-gvn DEPENDS ${objects})

# One ELF per slice: the harness with the slice's GPU values embedded
# (slice<i>/gvn_*.bin, written by the driver once the GPU has run), linked
# like the recipe's baremetal ELF.
add_custom_command(
  OUTPUT ${DESIGN_OUTPUT}/gvn/bert-base-uncased-emotion/npu/mlir-baremetal.o
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++17 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -fno-builtin -I${BUDDY_SOURCE}/llvm/mlir/include/mlir/ExecutionEngine -c ${RUNTIME_DIR}/services/mlir.cpp -o ${DESIGN_OUTPUT}/gvn/bert-base-uncased-emotion/npu/mlir-baremetal.o
  DEPENDS ${RUNTIME_DIR}/services/mlir.cpp
  VERBATIM
)
set(elfs)
foreach(slice ${slices})
  add_custom_command(
    OUTPUT ${slice}/slice.elf
    BYPRODUCTS ${slice}/npu.o
    COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++26 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -fno-builtin
              --embed-dir=${slice} -c ${CMAKE_CURRENT_SOURCE_DIR}/npu.cpp -o ${slice}/npu.o
    COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -specs=nano.specs -specs=nosys.specs -nostartfiles -Wl,-T,${BOOT_DIR}/device.ld ${DESIGN_DEFINES} -I${RUNTIME_DIR}/services -I${STACK_ROOT}/kernel/include ${BOOT_DIR}/start.S ${BOOT_DIR}/syscalls.c ${STACK_ROOT}/kernel/src/location.c ${RUNTIME_DIR}/trace/baremetal.c ${slice}/npu.o ${DESIGN_OUTPUT}/gvn/bert-base-uncased-emotion/npu/mlir-baremetal.o ${slice}/slice.o -lstdc++ -lm -lgcc -o ${slice}/slice.elf
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/npu.cpp
            ${slice}/gvn_in.bin
            ${slice}/gvn_table.bin
            ${slice}/gvn_expect.bin
            ${slice}/slice.o
            ${DESIGN_OUTPUT}/gvn/bert-base-uncased-emotion/npu/mlir-baremetal.o
            ${BOOT_DIR}/start.S
            ${BOOT_DIR}/syscalls.c
            ${BOOT_DIR}/device.ld
            ${STACK_ROOT}/kernel/src/location.c
            ${RUNTIME_DIR}/trace/baremetal.c
    VERBATIM
  )
  list(APPEND elfs ${slice}/slice.elf)
endforeach()
add_custom_target(bert-base-uncased-emotion-gvn-elf DEPENDS ${elfs})
