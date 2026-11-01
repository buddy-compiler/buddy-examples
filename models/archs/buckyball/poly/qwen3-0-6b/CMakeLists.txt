file(GLOB_RECURSE RECIPE_FRONTEND_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/frontend/*.py")

set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")
set(WEIGHTS "${MODEL_DIR}/qwen3-0-6b/output/qwen3-0-6b")
# The device compiler of poly: the device dialect, its placement and
# lowering, and lower-frontend-fake-op; built by the device stack. The program and
# the graphs of the reduce core, which has no hardware thread, are compiled
# with it.
set(DEVICE_COMPILER "${DESIGN_DEVICE_BUILD}/compiler/libdevice-compiler.so")
set(OPT ${DESIGN_COMPILER}/bin/buddy-opt --load-dialect-plugin=${DEVICE_COMPILER} --load-pass-plugin=${DEVICE_COMPILER})
# The core compiler of poly: the core dialect, the placement of the graphs of
# the poly cores on their attention and ffn threads and its lowering to the
# engine, and lower-frontend-fake-op. The graphs of the poly cores are compiled with
# it.
set(CORE_COMPILER "${DESIGN_DEVICE_BUILD}/core/compiler/libcore-compiler.so")
set(
  CORE_OPT
  ${DESIGN_COMPILER}/bin/buddy-opt
  --load-dialect-plugin=${CORE_COMPILER}
  --load-pass-plugin=${CORE_COMPILER}
)
set(ACC_OPT ${CORE_OPT} --buckyball-design-pb=${DESIGN_PB})
# The boot of every hart of the device: start.S, syscalls.c, device.ld.
set(BOOT_DIR "${DESIGN_REPO}/stack/runtime/boot")
list(TRANSFORM DESIGN_DEFS PREPEND "-D" OUTPUT_VARIABLE DESIGN_DEFINES)

# ===----------------------------------------------------------------------===
# Import Qwen3-0.6B for any number of tokens: the device program and the
# graphs of each class of core.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/program.mlir
         ${OUTPUT_DIR}/graph-reduce.mlir
         ${OUTPUT_DIR}/graph-poly.mlir
         ${OUTPUT_DIR}/prompt/pack-f32.data
         ${OUTPUT_DIR}/prompt/pack-i8.data
         ${OUTPUT_DIR}/prompt/inputs.data
         ${OUTPUT_DIR}/prompt/outputs.data
         ${OUTPUT_DIR}/prompt/sizes.txt
  BYPRODUCTS ${OUTPUT_DIR}/01-import.txt
             ${OUTPUT_DIR}/02-permuted.txt
             ${OUTPUT_DIR}/03-quanted.txt
             ${OUTPUT_DIR}/04-partitioned.txt
  COMMAND ${PYTHON} buddy-pytorch-importer.py ${DESIGN_PB}
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          ${RECIPE_FRONTEND_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${DESIGN_PB}
          ${MODEL_DIR}/qwen3-0-6b/model.py
          ${WEIGHTS}/config.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing the Qwen3-0.6B graph"
  VERBATIM
)
add_custom_target(qwen3-0-6b-import DEPENDS ${OUTPUT_DIR}/program.mlir)

# ===----------------------------------------------------------------------===
# The device program of Qwen3-0.6B placed on the cores and lowered to the
# device runtime (program_placed.mlir, program_bufferized.mlir,
# program_lowered.mlir).
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/program_lowered.mlir
  BYPRODUCTS ${OUTPUT_DIR}/program_placed.mlir
             ${OUTPUT_DIR}/program_bufferized.mlir
  COMMAND ${OPT} ${OUTPUT_DIR}/program.mlir -convert-linalg-to-device=design=${DESIGN_PB} -o ${OUTPUT_DIR}/program_placed.mlir
  COMMAND ${OPT} ${OUTPUT_DIR}/program_placed.mlir -convert-elementwise-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs" -o ${OUTPUT_DIR}/program_bufferized.mlir
  COMMAND ${OPT} ${OUTPUT_DIR}/program_bufferized.mlir -lower-device -o ${OUTPUT_DIR}/program_lowered.mlir
  DEPENDS ${OUTPUT_DIR}/program.mlir
          ${DEVICE_COMPILER}
          ${DESIGN_PB}
  COMMENT "Placing and lowering the Qwen3-0.6B program on poly"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The graphs of Qwen3-0.6B on the poly cores through the core level, common
# subexpressions merged with the conversion to linalg:
# convert-linalg-to-core puts their MXINT8 quantizations and products on the
# attention or ffn threads as subgraphs launched as early as their inputs
# allow (poly_core.mlir),
# lower-frontend-fake-op computes the rest of the frontend operations
# (poly_frontend.mlir), each segment's graph inlined into it is bufferized
# (poly_bufferized.mlir) and lower-core makes the transfers DDR buffers
# and the launches batches of asynchronous engine jobs
# (poly_memref.mlir). Each thread profile gets its own acc and bank lowering.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/poly_memref.mlir
  BYPRODUCTS ${OUTPUT_DIR}/poly_linalg.mlir
             ${OUTPUT_DIR}/poly_core.mlir
             ${OUTPUT_DIR}/poly_attention_acc.mlir
             ${OUTPUT_DIR}/poly_ffn_acc.mlir
             ${OUTPUT_DIR}/poly_attention_banks.mlir
             ${OUTPUT_DIR}/poly_attention_instructions.mlir
             ${OUTPUT_DIR}/poly_ffn_banks.mlir
             ${OUTPUT_DIR}/poly_ffn_instructions.mlir
             ${OUTPUT_DIR}/poly_addresses.mlir
             ${OUTPUT_DIR}/poly_frontend.mlir
             ${OUTPUT_DIR}/poly_bufferized.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/graph-poly.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith,cse))" -o ${OUTPUT_DIR}/poly_linalg.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/poly_linalg.mlir "-convert-linalg-to-core=design=${DESIGN_PB} core=poly" -o ${OUTPUT_DIR}/poly_core.mlir
  COMMAND ${ACC_OPT} ${OUTPUT_DIR}/poly_core.mlir
            --buckyball-target=attention -convert-linalg-to-acc -o ${OUTPUT_DIR}/poly_attention_acc.mlir
  COMMAND ${ACC_OPT} ${OUTPUT_DIR}/poly_attention_acc.mlir
            --buckyball-target=ffn -convert-linalg-to-acc -o ${OUTPUT_DIR}/poly_ffn_acc.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/poly_ffn_acc.mlir -lower-frontend-fake-op -o ${OUTPUT_DIR}/poly_frontend.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/poly_frontend.mlir -inline -eliminate-empty-tensors -convert-elementwise-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -drop-equivalent-buffer-results=modify-public-functions "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs" -ownership-based-buffer-deallocation -buffer-deallocation-simplification -bufferization-lower-deallocations -canonicalize -o ${OUTPUT_DIR}/poly_bufferized.mlir
  COMMAND ${ACC_OPT} ${OUTPUT_DIR}/poly_bufferized.mlir
            --buckyball-target=attention -assign-physical-banks -o ${OUTPUT_DIR}/poly_attention_banks.mlir
  COMMAND ${ACC_OPT} ${OUTPUT_DIR}/poly_attention_banks.mlir
            --buckyball-target=attention -lower-acc -o ${OUTPUT_DIR}/poly_attention_instructions.mlir
  COMMAND ${ACC_OPT} ${OUTPUT_DIR}/poly_attention_instructions.mlir
            --buckyball-target=ffn -assign-physical-banks -o ${OUTPUT_DIR}/poly_ffn_banks.mlir
  COMMAND ${ACC_OPT} ${OUTPUT_DIR}/poly_ffn_banks.mlir
            --buckyball-target=ffn -lower-acc -o ${OUTPUT_DIR}/poly_ffn_instructions.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/poly_ffn_instructions.mlir -expand-strided-metadata -lower-affine -canonicalize -o ${OUTPUT_DIR}/poly_addresses.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/poly_addresses.mlir "-lower-core=design=${DESIGN_PB} core=poly" -o ${OUTPUT_DIR}/poly_memref.mlir
  DEPENDS ${OUTPUT_DIR}/graph-poly.mlir
          ${CORE_COMPILER}
          ${DESIGN_PB}
  COMMENT "Lowering the Qwen3-0.6B graphs to the poly cores"
  VERBATIM
)
add_custom_target(qwen3-0-6b-core DEPENDS ${OUTPUT_DIR}/poly_memref.mlir)

# ===----------------------------------------------------------------------===
# The program of Qwen3-0.6B to RISC-V on the reduce core (program.ll).
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT program.o
  BYPRODUCTS ${OUTPUT_DIR}/program_llvm.mlir
             ${OUTPUT_DIR}/program.ll
  COMMAND ${OPT} ${OUTPUT_DIR}/program_lowered.mlir -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-func-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUTPUT_DIR}/program_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/program_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/program.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/program.ll -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -o program.o
  DEPENDS ${OUTPUT_DIR}/program_lowered.mlir
          ${DEVICE_COMPILER}
  COMMENT "Building the Qwen3-0.6B program for poly"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The graphs of Qwen3-0.6B on the reduce core, on its CPU in software:
# lower-frontend-fake-op computes the quantization (reduce_frontend.mlir),
# each graph inlined into its function writes its outputs into the buffers
# of the program (reduce_bufferized.mlir).
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT reduce.o
  BYPRODUCTS ${OUTPUT_DIR}/reduce_linalg.mlir
             ${OUTPUT_DIR}/reduce_frontend.mlir
             ${OUTPUT_DIR}/reduce_bufferized.mlir
             ${OUTPUT_DIR}/reduce_copies.mlir
             ${OUTPUT_DIR}/reduce_llvm.mlir
             ${OUTPUT_DIR}/reduce.ll
  COMMAND ${OPT} ${OUTPUT_DIR}/graph-reduce.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" -o ${OUTPUT_DIR}/reduce_linalg.mlir
  COMMAND ${OPT} ${OUTPUT_DIR}/reduce_linalg.mlir -lower-frontend-fake-op -o ${OUTPUT_DIR}/reduce_frontend.mlir
  COMMAND ${OPT} ${OUTPUT_DIR}/reduce_frontend.mlir -inline -eliminate-empty-tensors -convert-elementwise-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -drop-equivalent-buffer-results=modify-public-functions "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs" -ownership-based-buffer-deallocation -buffer-deallocation-simplification -bufferization-lower-deallocations -canonicalize -o ${OUTPUT_DIR}/reduce_bufferized.mlir
  COMMAND ${OPT} ${OUTPUT_DIR}/reduce_bufferized.mlir -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUTPUT_DIR}/reduce_copies.mlir
  COMMAND ${OPT} ${OUTPUT_DIR}/reduce_copies.mlir -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUTPUT_DIR}/reduce_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/reduce_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/reduce.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/reduce.ll -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -o reduce.o
  DEPENDS ${OUTPUT_DIR}/graph-reduce.mlir
          ${DEVICE_COMPILER}
  COMMENT "Building the Qwen3-0.6B graphs of the reduce core for poly"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The graphs of Qwen3-0.6B on the poly cores from the core level
# (poly_memref.mlir) to RISC-V. Until the acc level lowers the subgraphs to
# Ball instructions, the core CPU computes them in software when it records
# their jobs, and the threads run empty streams.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT poly.o
  BYPRODUCTS ${OUTPUT_DIR}/poly_copies.mlir
             ${OUTPUT_DIR}/poly_llvm.mlir
             ${OUTPUT_DIR}/poly.ll
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/poly_memref.mlir -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUTPUT_DIR}/poly_copies.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/poly_copies.mlir -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUTPUT_DIR}/poly_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/poly_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/poly.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/poly.ll -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -o poly.o
  DEPENDS ${OUTPUT_DIR}/poly_memref.mlir
          ${CORE_COMPILER}
  COMMENT "Building the Qwen3-0.6B graphs of the poly cores for poly"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The baremetal image of Qwen3-0.6B: the runner (decode.c, every hart enters
# it), the program and the graphs, the device runtime, the bindings of the
# cores and the loaders of their threads. The parameters, the inputs and
# the PyTorch outputs of the prompt are the model image (model-pack), at the
# model region of the DDR: <elf>-ddr.bin holds it at its offset in the DDR,
# zeros before it unwritten. Both go with the workloads of the device, where
# BEMU finds them.
# ===----------------------------------------------------------------------===
math(EXPR MODEL_OFFSET "${DESIGN_DEVICE_DDR_BYTES} - ${DESIGN_DEVICE_DDR_MODEL}")
math(EXPR MODEL_IMAGE "${DESIGN_DEVICE_DDR_BASE} + ${MODEL_OFFSET}" OUTPUT_FORMAT HEXADECIMAL)
set(ELF "${DESIGN_OUTPUT}/workloads/models/poly-qwen3-0-6b-baremetal")
add_custom_command(
  OUTPUT ${ELF}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DESIGN_OUTPUT}/workloads/models"
  COMMAND ${ELF_CC} -g -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany -specs=nano.specs -specs=nosys.specs -nostartfiles -Wl,-T,${BOOT_DIR}/device.ld ${DESIGN_DEFINES} -DWORKLOAD_REPORT_HART=${DESIGN_WORKLOAD_REPORT_HART} -DDEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART} -DMODEL_IMAGE=${MODEL_IMAGE}ULL -I${DESIGN_DEVICE_STACK}/runtime -I${DESIGN_REPO}/stack/runtime/program ${BOOT_DIR}/start.S ${BOOT_DIR}/syscalls.c ${CMAKE_CURRENT_SOURCE_DIR}/decode.c program.o reduce.o poly.o ${DESIGN_CORE_BUILD_reduce}/execute.o ${DESIGN_CORE_BUILD_poly}/execute.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o $<TARGET_FILE:poly-runtime> $<TARGET_FILE:runtime-program> $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:runtime> $<TARGET_FILE:engine> -lm -lgcc -o ${ELF}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/decode.c
          program.o
          reduce.o
          poly.o
          ${DESIGN_DEVICE_STACK}/runtime/device.h
          poly-runtime
          runtime-program
          runtime-baremetal
          runtime
          engine
  COMMENT "Linking Qwen3-0.6B for the poly device"
  VERBATIM
)
add_custom_command(
  OUTPUT ${ELF}-ddr.bin
  BYPRODUCTS model.img
  COMMAND ${CMAKE_BINARY_DIR}/model-pack model.img ${OUTPUT_DIR}/prompt pack-f32.data pack-i8.data inputs.data outputs.data sizes.txt
  COMMAND ${CMAKE_COMMAND} -E rm -f ${ELF}-ddr.bin
  COMMAND dd if=model.img of=${ELF}-ddr.bin bs=1M oflag=seek_bytes seek=${MODEL_OFFSET} conv=sparse status=none
  DEPENDS ${OUTPUT_DIR}/prompt/pack-f32.data
          ${OUTPUT_DIR}/prompt/pack-i8.data
          ${OUTPUT_DIR}/prompt/inputs.data
          ${OUTPUT_DIR}/prompt/outputs.data
          ${OUTPUT_DIR}/prompt/sizes.txt
          model-pack-tool
  COMMENT "Packing the Qwen3-0.6B prompt into the model region"
  VERBATIM
)
add_custom_target(qwen3-0-6b-model-build DEPENDS ${ELF} ${ELF}-ddr.bin)
