file(GLOB_RECURSE RECIPE_FRONTEND_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/frontend/*.py")

set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")
# The device compiler of piplus: the device dialect, its placement and
# lowering; built by the device stack. The program is compiled with it.
set(DEVICE_COMPILER "${DESIGN_DEVICE_BUILD}/compiler/libdevice-compiler.so")
set(OPT ${DESIGN_COMPILER}/bin/buddy-opt --load-dialect-plugin=${DEVICE_COMPILER} --load-pass-plugin=${DEVICE_COMPILER})
# The core compiler of piplus: the core dialect, its placement and lowering,
# and lower-frontend-fake-op; built by the device stack. The graphs are compiled
# with it.
set(CORE_COMPILER "${DESIGN_DEVICE_BUILD}/core/compiler/libcore-compiler.so")
set(
  CORE_OPT
  ${DESIGN_COMPILER}/bin/buddy-opt
  --load-dialect-plugin=${CORE_COMPILER}
  --load-pass-plugin=${CORE_COMPILER}
)
# The boot of every hart of the device: start.S, syscalls.c, device.ld.
set(BOOT_DIR "${DESIGN_REPO}/stack/runtime/boot")
list(TRANSFORM DESIGN_DEFS PREPEND "-D" OUTPUT_VARIABLE DESIGN_DEFINES)

# ===----------------------------------------------------------------------===
# Check the quantized program of smolvla-base on SO100 frames.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/calibration.json
  COMMAND ${PYTHON} buddy-pytorch-calibrate.py
  # The frontend sources the check imports: the layout of the weights
  # (frontend/quant/layout.py) changes no value and is not among them.
  DEPENDS ${STACK_ROOT}/compiler/frontend/quant.py
          ${CMAKE_CURRENT_SOURCE_DIR}/frontend/graphs.py
          ${CMAKE_CURRENT_SOURCE_DIR}/frontend/program.py
          ${CMAKE_CURRENT_SOURCE_DIR}/frontend/quant/graph.py
          ${CMAKE_CURRENT_SOURCE_DIR}/frontend/quant/reference.py
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-calibrate.py
          ${MODEL_DIR}/smolvla-base/model.py
          ${MODEL_DIR}/smolvla-base/observation.py
          ${MODEL_DIR}/smolvla-base/output/smolvla-base/config.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Checking the quantized smolvla-base actions for piplus"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# Import smolvla-base: its graphs, quantized, the device program, its inputs
# (one SO100 observation) and the actions of the quantized program in
# PyTorch; the program of the first denoising step alone, its inputs and its
# velocity in PyTorch.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/program.mlir
         ${OUTPUT_DIR}/graph-vision.mlir
         ${OUTPUT_DIR}/graph-prefix.mlir
         ${OUTPUT_DIR}/graph-step.mlir
         ${OUTPUT_DIR}/actions.data
         ${OUTPUT_DIR}/step_program.mlir
         ${OUTPUT_DIR}/step_velocity.data
  COMMAND ${PYTHON} buddy-pytorch-importer.py
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          ${RECIPE_FRONTEND_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${OUTPUT_DIR}/calibration.json
          ${MODEL_DIR}/smolvla-base/model.py
          ${MODEL_DIR}/smolvla-base/observation.py
          ${MODEL_DIR}/smolvla-base/output/smolvla-base/config.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing smolvla-base for piplus"
  VERBATIM
)
add_custom_target(smolvla-base-import DEPENDS ${OUTPUT_DIR}/program.mlir)

# ===----------------------------------------------------------------------===
# The device program of smolvla-base: its launches placed on the cores,
# transfers between them, lowered to the device runtime; on the control core.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT program.o
  BYPRODUCTS ${OUTPUT_DIR}/program_placed.mlir
             ${OUTPUT_DIR}/program_bufferized.mlir
             ${OUTPUT_DIR}/program_lowered.mlir
             ${OUTPUT_DIR}/program_llvm.mlir
             ${OUTPUT_DIR}/program.ll
  COMMAND ${OPT} ${OUTPUT_DIR}/program.mlir -convert-linalg-to-device=design=${DESIGN_PB} -o ${OUTPUT_DIR}/program_placed.mlir
  COMMAND ${OPT} ${OUTPUT_DIR}/program_placed.mlir -convert-elementwise-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs" -o ${OUTPUT_DIR}/program_bufferized.mlir
  COMMAND ${OPT} ${OUTPUT_DIR}/program_bufferized.mlir -lower-device -o ${OUTPUT_DIR}/program_lowered.mlir
  COMMAND ${OPT} ${OUTPUT_DIR}/program_lowered.mlir -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-func-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUTPUT_DIR}/program_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/program_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/program.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/program.ll -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -o program.o
  DEPENDS ${OUTPUT_DIR}/program.mlir
          ${DEVICE_COMPILER}
          ${DESIGN_PB}
  COMMENT "Building the smolvla-base program for piplus"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The program of the first denoising step alone, as the device program.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT step-program.o
  BYPRODUCTS ${OUTPUT_DIR}/step_program_placed.mlir
             ${OUTPUT_DIR}/step_program_bufferized.mlir
             ${OUTPUT_DIR}/step_program_lowered.mlir
             ${OUTPUT_DIR}/step_program_llvm.mlir
             ${OUTPUT_DIR}/step_program.ll
  COMMAND ${OPT} ${OUTPUT_DIR}/step_program.mlir -convert-linalg-to-device=design=${DESIGN_PB} -o ${OUTPUT_DIR}/step_program_placed.mlir
  COMMAND ${OPT} ${OUTPUT_DIR}/step_program_placed.mlir -convert-elementwise-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs" -o ${OUTPUT_DIR}/step_program_bufferized.mlir
  COMMAND ${OPT} ${OUTPUT_DIR}/step_program_bufferized.mlir -lower-device -o ${OUTPUT_DIR}/step_program_lowered.mlir
  COMMAND ${OPT} ${OUTPUT_DIR}/step_program_lowered.mlir -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-func-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUTPUT_DIR}/step_program_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/step_program_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/step_program.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/step_program.ll -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -o step-program.o
  DEPENDS ${OUTPUT_DIR}/step_program.mlir
          ${DEVICE_COMPILER}
          ${DESIGN_PB}
  COMMENT "Building the smolvla-base step program for piplus"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The graphs of smolvla-base on their cores (vision and prefix on the
# dense cores, the step on the action core), common subexpressions merged
# with the conversion to linalg: convert-linalg-to-core puts the MXINT8
# quantization and products on the core's hardware thread as subgraphs
# launched as early as their inputs allow (<graph>_core.mlir),
# convert-linalg-to-acc maps MXINT8 products and quantization to bank loops;
# bank assignment and lower-acc emit Ball instructions. lower-core records
# their affine loops for the engine, retaining staging memory until completion.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT vision.o
  BYPRODUCTS ${OUTPUT_DIR}/vision_linalg.mlir
             ${OUTPUT_DIR}/vision_core.mlir
             ${OUTPUT_DIR}/vision_acc.mlir
             ${OUTPUT_DIR}/vision_banks.mlir
             ${OUTPUT_DIR}/vision_instructions.mlir
             ${OUTPUT_DIR}/vision_addresses.mlir
             ${OUTPUT_DIR}/vision_frontend.mlir
             ${OUTPUT_DIR}/vision_bufferized.mlir
             ${OUTPUT_DIR}/vision_memref.mlir
             ${OUTPUT_DIR}/vision_copies.mlir
             ${OUTPUT_DIR}/vision_llvm.mlir
             ${OUTPUT_DIR}/vision.ll
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/graph-vision.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith,cse))" -o ${OUTPUT_DIR}/vision_linalg.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/vision_linalg.mlir "-convert-linalg-to-core=design=${DESIGN_PB} core=dense" -o ${OUTPUT_DIR}/vision_core.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/vision_core.mlir
            -convert-linalg-to-acc
            -o ${OUTPUT_DIR}/vision_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/vision_acc.mlir
            -lower-frontend-fake-op
            -o ${OUTPUT_DIR}/vision_frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/vision_frontend.mlir
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
            -drop-equivalent-buffer-results "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -canonicalize
            -o ${OUTPUT_DIR}/vision_bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/vision_bufferized.mlir
            -assign-physical-banks
            -o ${OUTPUT_DIR}/vision_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/vision_banks.mlir
            -lower-acc
            -o ${OUTPUT_DIR}/vision_instructions.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/vision_instructions.mlir -expand-strided-metadata -lower-affine -canonicalize -o ${OUTPUT_DIR}/vision_addresses.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/vision_addresses.mlir "-lower-core=design=${DESIGN_PB} core=dense" -o ${OUTPUT_DIR}/vision_memref.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/vision_memref.mlir -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUTPUT_DIR}/vision_copies.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/vision_copies.mlir -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts -ensure-debug-info-scope-on-llvm-func -o ${OUTPUT_DIR}/vision_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/vision_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/vision.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/vision.ll -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -o vision.o
  DEPENDS ${OUTPUT_DIR}/graph-vision.mlir
          ${DESIGN_COMPILER}/bin/buddy-opt
          ${DESIGN_THREAD_BUILD_dense}/libdense-compiler.so
          ${CORE_COMPILER}
          ${DESIGN_PB}
  COMMENT "Building the smolvla-base vision graph for piplus"
  VERBATIM
)

add_custom_command(
  OUTPUT prefix.o
  BYPRODUCTS ${OUTPUT_DIR}/prefix_linalg.mlir
             ${OUTPUT_DIR}/prefix_core.mlir
             ${OUTPUT_DIR}/prefix_acc.mlir
             ${OUTPUT_DIR}/prefix_banks.mlir
             ${OUTPUT_DIR}/prefix_instructions.mlir
             ${OUTPUT_DIR}/prefix_addresses.mlir
             ${OUTPUT_DIR}/prefix_frontend.mlir
             ${OUTPUT_DIR}/prefix_bufferized.mlir
             ${OUTPUT_DIR}/prefix_memref.mlir
             ${OUTPUT_DIR}/prefix_copies.mlir
             ${OUTPUT_DIR}/prefix_llvm.mlir
             ${OUTPUT_DIR}/prefix.ll
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/graph-prefix.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith,cse))" -o ${OUTPUT_DIR}/prefix_linalg.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/prefix_linalg.mlir "-convert-linalg-to-core=design=${DESIGN_PB} core=dense" -o ${OUTPUT_DIR}/prefix_core.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/prefix_core.mlir
            -convert-linalg-to-acc
            -o ${OUTPUT_DIR}/prefix_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/prefix_acc.mlir
            -lower-frontend-fake-op
            -o ${OUTPUT_DIR}/prefix_frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/prefix_frontend.mlir
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
            -drop-equivalent-buffer-results "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -canonicalize
            -o ${OUTPUT_DIR}/prefix_bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/prefix_bufferized.mlir
            -assign-physical-banks
            -o ${OUTPUT_DIR}/prefix_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/prefix_banks.mlir
            -lower-acc
            -o ${OUTPUT_DIR}/prefix_instructions.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/prefix_instructions.mlir -expand-strided-metadata -lower-affine -canonicalize -o ${OUTPUT_DIR}/prefix_addresses.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/prefix_addresses.mlir "-lower-core=design=${DESIGN_PB} core=dense" -o ${OUTPUT_DIR}/prefix_memref.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/prefix_memref.mlir -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUTPUT_DIR}/prefix_copies.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/prefix_copies.mlir -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts -ensure-debug-info-scope-on-llvm-func -o ${OUTPUT_DIR}/prefix_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/prefix_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/prefix.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/prefix.ll -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -o prefix.o
  DEPENDS ${OUTPUT_DIR}/graph-prefix.mlir
          ${DESIGN_COMPILER}/bin/buddy-opt
          ${DESIGN_THREAD_BUILD_dense}/libdense-compiler.so
          ${CORE_COMPILER}
          ${DESIGN_PB}
  COMMENT "Building the smolvla-base prefix graph for piplus"
  VERBATIM
)

add_custom_command(
  OUTPUT step.o
  BYPRODUCTS ${OUTPUT_DIR}/step_linalg.mlir
             ${OUTPUT_DIR}/step_core.mlir
             ${OUTPUT_DIR}/step_acc.mlir
             ${OUTPUT_DIR}/step_banks.mlir
             ${OUTPUT_DIR}/step_instructions.mlir
             ${OUTPUT_DIR}/step_addresses.mlir
             ${OUTPUT_DIR}/step_frontend.mlir
             ${OUTPUT_DIR}/step_bufferized.mlir
             ${OUTPUT_DIR}/step_memref.mlir
             ${OUTPUT_DIR}/step_copies.mlir
             ${OUTPUT_DIR}/step_llvm.mlir
             ${OUTPUT_DIR}/step.ll
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/graph-step.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith,cse))" -o ${OUTPUT_DIR}/step_linalg.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/step_linalg.mlir "-convert-linalg-to-core=design=${DESIGN_PB} core=action" -o ${OUTPUT_DIR}/step_core.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/step_core.mlir
            -convert-linalg-to-acc
            -o ${OUTPUT_DIR}/step_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/step_acc.mlir
            -lower-frontend-fake-op
            -o ${OUTPUT_DIR}/step_frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/step_frontend.mlir
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
            -drop-equivalent-buffer-results "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -canonicalize
            -o ${OUTPUT_DIR}/step_bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/step_bufferized.mlir
            -assign-physical-banks
            -o ${OUTPUT_DIR}/step_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/step_banks.mlir
            -lower-acc
            -o ${OUTPUT_DIR}/step_instructions.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/step_instructions.mlir -expand-strided-metadata -lower-affine -canonicalize -o ${OUTPUT_DIR}/step_addresses.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/step_addresses.mlir "-lower-core=design=${DESIGN_PB} core=action" -o ${OUTPUT_DIR}/step_memref.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/step_memref.mlir -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUTPUT_DIR}/step_copies.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/step_copies.mlir -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts -ensure-debug-info-scope-on-llvm-func -o ${OUTPUT_DIR}/step_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/step_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/step.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/step.ll -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -o step.o
  DEPENDS ${OUTPUT_DIR}/graph-step.mlir
          ${DESIGN_COMPILER}/bin/buddy-opt
          ${DESIGN_THREAD_BUILD_action}/libaction-compiler.so
          ${CORE_COMPILER}
          ${DESIGN_PB}
  COMMENT "Building the smolvla-base step graph for piplus"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The baremetal image of smolvla-base: the runner (every hart enters it),
# the program and the graphs, the device runtime, the bindings of the cores
# and the loaders of their threads (a core with threads starts its engine for
# every job); the parameters, inputs and reference actions embedded. It goes with
# the workloads of the device, where BEMU finds it.
# ===----------------------------------------------------------------------===
set(ELF "${DESIGN_OUTPUT}/workloads/models/piplus-smolvla-base-baremetal")
add_custom_command(
  OUTPUT ${ELF}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DESIGN_OUTPUT}/workloads/models"
  COMMAND ${ELF_CC} -g -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany -specs=nano.specs -specs=nosys.specs -nostartfiles -Wl,-T,${BOOT_DIR}/device.ld ${DESIGN_DEFINES} -DWORKLOAD_REPORT_HART=${DESIGN_WORKLOAD_REPORT_HART} -DDEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART} -I${DESIGN_DEVICE_STACK}/runtime -I${DESIGN_REPO}/stack/runtime/program
            --embed-dir=${OUTPUT_DIR} ${BOOT_DIR}/start.S ${BOOT_DIR}/syscalls.c ${CMAKE_CURRENT_SOURCE_DIR}/runner.c program.o vision.o prefix.o step.o ${DESIGN_CORE_BUILD_control}/execute.o ${DESIGN_CORE_BUILD_dense}/execute.o ${DESIGN_CORE_BUILD_action}/execute.o ${DESIGN_THREAD_BUILD_dense}/loader.o ${DESIGN_THREAD_BUILD_action}/loader.o $<TARGET_FILE:piplus-runtime> $<TARGET_FILE:runtime-program> $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:runtime> $<TARGET_FILE:engine> -lm -lgcc -o ${ELF}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/runner.c
          program.o
          vision.o
          prefix.o
          step.o
          ${OUTPUT_DIR}/actions.data
          ${DESIGN_DEVICE_STACK}/runtime/device.h
          piplus-runtime
          runtime-program
          runtime-baremetal
          runtime
          engine
  COMMENT "Linking smolvla-base for the piplus device"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The baremetal image of the first denoising step of smolvla-base alone: the
# runner (runner-step.c), the step program and the step graph, as the image
# of the policy; the parameters of the step, its inputs (from the quantized
# program on the observation) and its reference velocity embedded.
# ===----------------------------------------------------------------------===
set(STEP_ELF "${DESIGN_OUTPUT}/workloads/models/piplus-smolvla-base-step-baremetal")
add_custom_command(
  OUTPUT ${STEP_ELF}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DESIGN_OUTPUT}/workloads/models"
  COMMAND ${ELF_CC} -g -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany -specs=nano.specs -specs=nosys.specs -nostartfiles -Wl,-T,${BOOT_DIR}/device.ld ${DESIGN_DEFINES} -DWORKLOAD_REPORT_HART=${DESIGN_WORKLOAD_REPORT_HART} -DDEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART} -I${DESIGN_DEVICE_STACK}/runtime -I${DESIGN_REPO}/stack/runtime/program
            --embed-dir=${OUTPUT_DIR} ${BOOT_DIR}/start.S ${BOOT_DIR}/syscalls.c ${CMAKE_CURRENT_SOURCE_DIR}/runner-step.c step-program.o step.o ${DESIGN_CORE_BUILD_control}/execute.o ${DESIGN_CORE_BUILD_dense}/execute.o ${DESIGN_CORE_BUILD_action}/execute.o ${DESIGN_THREAD_BUILD_dense}/loader.o ${DESIGN_THREAD_BUILD_action}/loader.o $<TARGET_FILE:piplus-runtime> $<TARGET_FILE:runtime-program> $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:runtime> $<TARGET_FILE:engine> -lm -lgcc -o ${STEP_ELF}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/runner-step.c
          step-program.o
          step.o
          ${OUTPUT_DIR}/step_velocity.data
          ${DESIGN_DEVICE_STACK}/runtime/device.h
          piplus-runtime
          runtime-program
          runtime-baremetal
          runtime
          engine
  COMMENT "Linking the first smolvla-base step for the piplus device"
  VERBATIM
)
add_custom_target(smolvla-base-model-build DEPENDS ${ELF} ${STEP_ELF})
