set(NATIVE_DIR "${OUTPUT_DIR}/native")
set(POLY_BUILD "${CMAKE_CURRENT_BINARY_DIR}/..")
set(NATIVE_PROGRAMS)
set(NATIVE_SOURCES "${NATIVE_DIR}/native-bindings.c" "${NATIVE_DIR}/native.json")
foreach(index IN LISTS DESIGN_DEVICE_HARTS_${DESIGN_HART_HART_poly}_THREAD_INDICES)
  set(thread ${DESIGN_THREADS_${index}_LOCAL_ID})
  list(APPEND NATIVE_SOURCES "${NATIVE_DIR}/thread-${thread}.mlir" "${NATIVE_DIR}/thread-${thread}.S")
endforeach()

add_custom_command(
  OUTPUT "${NATIVE_DIR}/host.mlir"
         ${NATIVE_SOURCES}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${NATIVE_DIR}"
  COMMAND ${DESIGN_HART_OPT_poly} ${OUTPUT_DIR}/poly_addresses.mlir "-lower-hart-native=design=${DESIGN_PB} hart=poly output-dir=${NATIVE_DIR}" -o "${NATIVE_DIR}/host.mlir"
  DEPENDS ${OUTPUT_DIR}/poly_memref.mlir
          ${DESIGN_HART_PLUGIN_poly}
          ${DESIGN_PB}
  COMMENT "Compiling Qwen3-8B thread programs and host preparation"
  VERBATIM
)

foreach(index IN LISTS DESIGN_DEVICE_HARTS_${DESIGN_HART_HART_poly}_THREAD_INDICES)
  set(thread ${DESIGN_THREADS_${index}_LOCAL_ID})
  add_custom_command(
    OUTPUT "${NATIVE_DIR}/thread-${thread}.bin"
    BYPRODUCTS "${NATIVE_DIR}/thread-${thread}-llvm.mlir"
               "${NATIVE_DIR}/thread-${thread}.ll"
               "${NATIVE_DIR}/thread-${thread}-opt.ll"
               "${NATIVE_DIR}/thread-${thread}.o"
               "${NATIVE_DIR}/thread-${thread}-start.o"
               "${NATIVE_DIR}/thread-${thread}.elf"
    COMMAND ${DESIGN_HART_OPT_poly} "${NATIVE_DIR}/thread-${thread}.mlir" -expand-strided-metadata -lower-affine -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts -o "${NATIVE_DIR}/thread-${thread}-llvm.mlir"
    COMMAND ${DESIGN_COMPILER}/bin/buddy-translate "${NATIVE_DIR}/thread-${thread}-llvm.mlir"
              --buddy-to-llvmir -o "${NATIVE_DIR}/thread-${thread}.ll"
    COMMAND ${LLVM_OPT} "${NATIVE_DIR}/thread-${thread}.ll" -mtriple=riscv64 -mattr=+xbuckyball,+m "-passes=default<O2>" -S -o "${NATIVE_DIR}/thread-${thread}-opt.ll"
    COMMAND ${DESIGN_COMPILER}/bin/buddy-llc "${NATIVE_DIR}/thread-${thread}-opt.ll" -filetype=obj -mtriple=riscv64 -mattr=+xbuckyball,+m -target-abi=lp64 -code-model=medium -riscv-disable-using-constant-pool-for-large-ints -stack-size-section -O2 -o "${NATIVE_DIR}/thread-${thread}.o"
    COMMAND ${ELF_CC} -c -march=rv64im -mabi=lp64 "${NATIVE_DIR}/thread-${thread}.S" -o "${NATIVE_DIR}/thread-${thread}-start.o"
    COMMAND ${ELF_CC} -nostdlib -static -march=rv64im -mabi=lp64 "-Wl,-T,${DESIGN_REPO}/stack/kernel/program/code.ld" -Wl,-e,bb_native_start_0 -Wl,--no-relax "${NATIVE_DIR}/thread-${thread}-start.o" "${NATIVE_DIR}/thread-${thread}.o" -o "${NATIVE_DIR}/thread-${thread}.elf"
    COMMAND ${ELF_OBJCOPY} -O binary -j .text "${NATIVE_DIR}/thread-${thread}.elf" "${NATIVE_DIR}/thread-${thread}.bin"
    DEPENDS "${NATIVE_DIR}/host.mlir"
            "${NATIVE_DIR}/thread-${thread}.mlir"
            "${NATIVE_DIR}/thread-${thread}.S"
            ${DESIGN_HART_PLUGIN_poly}
            ${DESIGN_REPO}/stack/kernel/program/code.ld
    COMMENT "Building Qwen3-8B program for thread ${thread}"
    VERBATIM
  )
  list(APPEND NATIVE_PROGRAMS "${NATIVE_DIR}/thread-${thread}.bin")
endforeach()

add_custom_command(
  OUTPUT "${NATIVE_DIR}/bindings.o"
  COMMAND ${ELF_CC} -std=c2x -O2 -fPIC -ffreestanding -msmall-data-limit=0 -march=rv64gc -mabi=lp64d -mcmodel=medany -I${DESIGN_REPO}/stack/kernel/include -c "${NATIVE_DIR}/native-bindings.c" -o "${NATIVE_DIR}/bindings.o"
  DEPENDS "${NATIVE_DIR}/native-bindings.c"
          ${NATIVE_PROGRAMS}
          ${DESIGN_REPO}/stack/kernel/include/kernel.h
          ${DESIGN_REPO}/stack/kernel/include/thread.h
  VERBATIM
)

add_custom_command(
  OUTPUT "${NATIVE_DIR}/poly.o"
  BYPRODUCTS "${NATIVE_DIR}/copies.mlir"
             "${NATIVE_DIR}/llvm.mlir"
             "${NATIVE_DIR}/host.ll"
             "${NATIVE_DIR}/host-opt.ll"
  COMMAND ${DESIGN_HART_OPT_poly} "${NATIVE_DIR}/host.mlir" -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o "${NATIVE_DIR}/copies.mlir"
  COMMAND ${DESIGN_HART_OPT_poly} "${NATIVE_DIR}/copies.mlir" -convert-linalg-to-affine-loops -affine-loop-invariant-code-motion -expand-strided-metadata -lower-affine -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm -reconcile-unrealized-casts -o "${NATIVE_DIR}/llvm.mlir"
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate "${NATIVE_DIR}/llvm.mlir"
            --buddy-to-llvmir -o "${NATIVE_DIR}/host.ll"
  COMMAND ${LLVM_OPT} "${NATIVE_DIR}/host.ll" -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=default<O2>" -S -o "${NATIVE_DIR}/host-opt.ll"
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc "${NATIVE_DIR}/host-opt.ll" -filetype=obj -relocation-model=pic -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -o "${NATIVE_DIR}/poly.o"
  DEPENDS "${NATIVE_DIR}/host.mlir"
          ${DESIGN_HART_PLUGIN_poly}
  VERBATIM
)

set(NATIVE_ELF "${DESIGN_OUTPUT}/workloads/models/poly-qwen3-8b-native-baremetal")
add_custom_command(
  OUTPUT "${NATIVE_ELF}"
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany -specs=nano.specs -specs=nosys.specs -nostartfiles "-Wl,-T,${BOOT_DIR}/device.ld" ${DESIGN_DEFINES} -DWORKLOAD_REPORT_HART=${DESIGN_WORKLOAD_REPORT_HART} -DDEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART} -DMODEL_IMAGE=${MODEL_IMAGE}ULL -I${DESIGN_DEVICE_STACK}/runtime -I${DESIGN_REPO}/stack/runtime/program ${BOOT_DIR}/start.S ${BOOT_DIR}/syscalls.c "${POLY_BUILD}/baremetal.o" "${POLY_BUILD}/program.o" "${POLY_BUILD}/reduce.o" "${NATIVE_DIR}/poly.o" "${NATIVE_DIR}/bindings.o" "${POLY_BUILD}/operators.embed.o" "${POLY_BUILD}/reduce_operators.embed.o" ${DESIGN_HART_BUILD_${REDUCE_HART}}/execute.o ${DESIGN_HART_BUILD_poly}/execute.o $<TARGET_FILE:poly-runtime> $<TARGET_FILE:runtime-program> $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:runtime> $<TARGET_FILE:kernel> -lm -lgcc -o "${NATIVE_ELF}"
  DEPENDS ${ELF}
          "${NATIVE_DIR}/poly.o"
          "${NATIVE_DIR}/bindings.o"
          ${BOOT_DIR}/device.ld
          ${BOOT_DIR}/start.S
          ${BOOT_DIR}/syscalls.c
          poly-runtime
          runtime-program
          runtime-baremetal
          runtime
          kernel
  COMMENT "Linking native Qwen3-8B with compiler-generated kernel bindings"
  VERBATIM
)
add_custom_command(
  OUTPUT "${NATIVE_ELF}-ddr.bin"
  COMMAND ${CMAKE_COMMAND} -E create_symlink "${ELF}-ddr.bin" "${NATIVE_ELF}-ddr.bin"
  DEPENDS "${ELF}-ddr.bin"
  VERBATIM
)
add_custom_target(qwen3-8b-native-model-build DEPENDS "${NATIVE_ELF}" "${NATIVE_ELF}-ddr.bin")
