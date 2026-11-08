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
          ${STACK_ROOT}/flake.lock
          ${STACK_ROOT}/scripts/nix/riscv.nix
          ${STACK_ROOT}/result/lib/libnewlib-reference.so
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
         ${OUTPUT_DIR}/vision-arg0.data
         ${OUTPUT_DIR}/vision-arg1.data
         ${OUTPUT_DIR}/prefix-arg0.data
         ${OUTPUT_DIR}/prefix-arg1.data
         ${OUTPUT_DIR}/step-arg0.data
         ${OUTPUT_DIR}/step-arg1.data
         ${OUTPUT_DIR}/images.data
         ${OUTPUT_DIR}/lang_tokens.data
         ${OUTPUT_DIR}/lang_masks.data
         ${OUTPUT_DIR}/state.data
         ${OUTPUT_DIR}/noise.data
         ${OUTPUT_DIR}/step_time.data
         ${OUTPUT_DIR}/step_pad_masks.data
         ${OUTPUT_DIR}/step_cache.data
         ${OUTPUT_DIR}/observation.data
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
  COMMAND ${OPT} ${OUTPUT_DIR}/program_placed.mlir -convert-elementwise-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs hoist-dynamic-allocs" -o ${OUTPUT_DIR}/program_bufferized.mlir
  COMMAND ${OPT} ${OUTPUT_DIR}/program_bufferized.mlir -canonicalize -cse -lower-device -o ${OUTPUT_DIR}/program_lowered.mlir
  COMMAND ${OPT} ${OUTPUT_DIR}/program_lowered.mlir -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts -o ${OUTPUT_DIR}/program_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/program_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/program.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/program.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -o program.o
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
  COMMAND ${OPT} ${OUTPUT_DIR}/step_program_placed.mlir -convert-elementwise-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs hoist-dynamic-allocs" -o ${OUTPUT_DIR}/step_program_bufferized.mlir
  COMMAND ${OPT} ${OUTPUT_DIR}/step_program_bufferized.mlir -canonicalize -cse -lower-device -o ${OUTPUT_DIR}/step_program_lowered.mlir
  COMMAND ${OPT} ${OUTPUT_DIR}/step_program_lowered.mlir -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts -o ${OUTPUT_DIR}/step_program_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/step_program_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/step_program.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/step_program.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -o step-program.o
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
            -inline
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
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/vision_copies.mlir -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm -reconcile-unrealized-casts -ensure-debug-info-scope-on-llvm-func -o ${OUTPUT_DIR}/vision_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/vision_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/vision.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/vision.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -o vision.o
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
            -inline
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
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/prefix_copies.mlir -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm -reconcile-unrealized-casts -ensure-debug-info-scope-on-llvm-func -o ${OUTPUT_DIR}/prefix_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/prefix_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/prefix.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/prefix.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -o prefix.o
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
            -inline
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
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/step_copies.mlir -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm -reconcile-unrealized-casts -ensure-debug-info-scope-on-llvm-func -o ${OUTPUT_DIR}/step_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/step_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/step.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/step.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -o step.o
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
math(EXPR MODEL_OFFSET "${DESIGN_DEVICE_DDR_BYTES} - ${DESIGN_DEVICE_DDR_MODEL}")
math(EXPR MODEL_IMAGE "${DESIGN_DEVICE_DDR_BASE} + ${MODEL_OFFSET}" OUTPUT_FORMAT HEXADECIMAL)
file(READ ${MODEL_DIR}/smolvla-base/output/smolvla-base/config.json MODEL_CONFIG)
string(JSON CHUNK_SIZE GET "${MODEL_CONFIG}" chunk_size)
string(JSON ACTION_DIM GET "${MODEL_CONFIG}" max_action_dim)
math(EXPR ACTION_ELEMENTS "${CHUNK_SIZE} * ${ACTION_DIM}")
set(ELF "${DESIGN_OUTPUT}/workloads/models/piplus-smolvla-base-baremetal")
add_custom_command(
  OUTPUT ${ELF}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DESIGN_OUTPUT}/workloads/models"
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++17 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -fno-exceptions -fno-rtti ${DESIGN_DEFINES} -DDEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART} -DMODEL_IMAGE=${MODEL_IMAGE}ULL -DACTION_ELEMENTS=${ACTION_ELEMENTS} -I${LIB_DIR} -I${DESIGN_DEVICE_STACK}/runtime -I${DESIGN_REPO}/stack/runtime/program -c ${CMAKE_CURRENT_SOURCE_DIR}/baremetal.cpp -o baremetal.o
  COMMAND ${ELF_CC} -g -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany -specs=nano.specs -specs=nosys.specs -nostartfiles -Wl,-T,${BOOT_DIR}/device.ld ${DESIGN_DEFINES} -DWORKLOAD_REPORT_HART=${DESIGN_WORKLOAD_REPORT_HART} -DDEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART} -I${DESIGN_DEVICE_STACK}/runtime -I${DESIGN_REPO}/stack/runtime/program ${BOOT_DIR}/start.S ${BOOT_DIR}/syscalls.c baremetal.o program.o vision.o prefix.o step.o ${DESIGN_CORE_BUILD_control}/execute.o ${DESIGN_CORE_BUILD_dense}/execute.o ${DESIGN_CORE_BUILD_action}/execute.o ${DESIGN_THREAD_BUILD_dense}/loader.o ${DESIGN_THREAD_BUILD_action}/loader.o $<TARGET_FILE:piplus-runtime> $<TARGET_FILE:runtime-program> $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:runtime> $<TARGET_FILE:engine> -lm -lgcc -o ${ELF}
  DEPENDS ${BOOT_DIR}/start.S
          ${BOOT_DIR}/syscalls.c
          ${BOOT_DIR}/device.ld
          ${CMAKE_CURRENT_SOURCE_DIR}/baremetal.cpp
          model.h
          ${LIB_DIR}/ModelImage.h
          ${LIB_DIR}/testutils.h
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
# runner (baremetal-step.cpp), the step program and the step graph, as the image
# of the policy; the parameters of the step, its inputs (from the quantized
# program on the observation) and its reference velocity embedded.
# ===----------------------------------------------------------------------===
set(STEP_ELF "${DESIGN_OUTPUT}/workloads/models/piplus-smolvla-base-step-baremetal")
add_custom_command(
  OUTPUT ${STEP_ELF}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DESIGN_OUTPUT}/workloads/models"
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++20 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -fno-exceptions -fno-rtti ${DESIGN_DEFINES} -DDEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART} -I${LIB_DIR} -I${DESIGN_DEVICE_STACK}/runtime -I${DESIGN_REPO}/stack/runtime/program
            --embed-dir=${OUTPUT_DIR} -c ${CMAKE_CURRENT_SOURCE_DIR}/baremetal-step.cpp -o baremetal-step.o
  COMMAND ${ELF_CC} -g -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany -specs=nano.specs -specs=nosys.specs -nostartfiles -Wl,-T,${BOOT_DIR}/device.ld ${DESIGN_DEFINES} -DWORKLOAD_REPORT_HART=${DESIGN_WORKLOAD_REPORT_HART} -DDEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART} -I${DESIGN_DEVICE_STACK}/runtime -I${DESIGN_REPO}/stack/runtime/program
            --embed-dir=${OUTPUT_DIR} ${BOOT_DIR}/start.S ${BOOT_DIR}/syscalls.c baremetal-step.o step-program.o step.o ${DESIGN_CORE_BUILD_control}/execute.o ${DESIGN_CORE_BUILD_dense}/execute.o ${DESIGN_CORE_BUILD_action}/execute.o ${DESIGN_THREAD_BUILD_dense}/loader.o ${DESIGN_THREAD_BUILD_action}/loader.o $<TARGET_FILE:piplus-runtime> $<TARGET_FILE:runtime-program> $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:runtime> $<TARGET_FILE:engine> -lm -lgcc -o ${STEP_ELF}
  DEPENDS ${BOOT_DIR}/start.S
          ${BOOT_DIR}/syscalls.c
          ${BOOT_DIR}/device.ld
          ${CMAKE_CURRENT_SOURCE_DIR}/baremetal-step.cpp
          ${LIB_DIR}/testutils.h
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

add_library(smolvla-base-buckyball-piplus-model SHARED EXCLUDE_FROM_ALL entry.cpp program.o vision.o prefix.o step.o)
target_link_libraries(smolvla-base-buckyball-piplus-model
  PRIVATE piplus-runtime-linux
)
target_link_options(smolvla-base-buckyball-piplus-model PRIVATE "LINKER:--no-undefined")
set_target_properties(smolvla-base-buckyball-piplus-model
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN;/lib"
)
get_filename_component(FLATBUFFERS_HEADER "${TOOLCHAIN}/include/flatbuffers/flatbuffers.h" REALPATH)
get_filename_component(FLATBUFFERS_INCLUDE "${FLATBUFFERS_HEADER}/../.." ABSOLUTE)
add_library(smolvla-base-buckyball-piplus-runner SHARED EXCLUDE_FROM_ALL runner.cpp)
target_include_directories(smolvla-base-buckyball-piplus-runner
  PRIVATE ${LIB_DIR}
          ${RUNTIME_DIR}/services
          ${FLATBUFFERS_INCLUDE}
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/runtime/include
          ${DESIGN_COMPILER}/runtime/include
)
target_compile_definitions(smolvla-base-buckyball-piplus-runner
  PRIVATE ACTION_ELEMENTS=${ACTION_ELEMENTS}
)
target_link_libraries(smolvla-base-buckyball-piplus-runner
  PRIVATE dl
)
set_target_properties(smolvla-base-buckyball-piplus-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN;/lib"
)
configure_file(smolvla-base.rhal.mlir.in ${OUTPUT_DIR}/smolvla-base.rhal.mlir @ONLY)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/smolvla-base.rax
  COMMAND ${DESIGN_COMPILER}/bin/rax-pack ${OUTPUT_DIR}/smolvla-base.rhal.mlir
          --embed-payload
          -o ${OUTPUT_DIR}/smolvla-base.rax
  DEPENDS ${OUTPUT_DIR}/smolvla-base.rhal.mlir
          ${DESIGN_COMPILER}/bin/rax-pack
          smolvla-base-buckyball-piplus-model
          smolvla-base-buckyball-piplus-runner
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)
file(WRITE ${OUTPUT_DIR}/run.args "--model\nsmolvla-base.rax\n--input\nobservation.data\n--output\nactions.f32\n")
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/model.img
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}/ddr
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${OUTPUT_DIR}/vision-arg0.data ${OUTPUT_DIR}/ddr/vision-arg0.data
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${OUTPUT_DIR}/vision-arg1.data ${OUTPUT_DIR}/ddr/vision-arg1.data
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${OUTPUT_DIR}/prefix-arg0.data ${OUTPUT_DIR}/ddr/prefix-arg0.data
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${OUTPUT_DIR}/prefix-arg1.data ${OUTPUT_DIR}/ddr/prefix-arg1.data
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${OUTPUT_DIR}/step-arg0.data ${OUTPUT_DIR}/ddr/step-arg0.data
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${OUTPUT_DIR}/step-arg1.data ${OUTPUT_DIR}/ddr/step-arg1.data
  COMMAND ${CMAKE_BINARY_DIR}/model-pack ${OUTPUT_DIR}/model.img ${OUTPUT_DIR} smolvla-base.rax
            --mapped run.args observation.data actions.data
            --mapped ddr/vision-arg0.data
            --mapped ddr/vision-arg1.data
            --mapped ddr/prefix-arg0.data
            --mapped ddr/prefix-arg1.data
            --mapped ddr/step-arg0.data
            --mapped ddr/step-arg1.data
  DEPENDS ${OUTPUT_DIR}/smolvla-base.rax
          ${OUTPUT_DIR}/run.args
          ${OUTPUT_DIR}/observation.data
          ${OUTPUT_DIR}/actions.data
          ${OUTPUT_DIR}/vision-arg0.data
          ${OUTPUT_DIR}/vision-arg1.data
          ${OUTPUT_DIR}/prefix-arg0.data
          ${OUTPUT_DIR}/prefix-arg1.data
          ${OUTPUT_DIR}/step-arg0.data
          ${OUTPUT_DIR}/step-arg1.data
          model-pack-tool
  VERBATIM
)
add_custom_command(
  OUTPUT ${ELF}-ddr.bin
  COMMAND ${CMAKE_COMMAND} -E rm -f ${ELF}-ddr.bin
  COMMAND dd if=${OUTPUT_DIR}/model.img of=${ELF}-ddr.bin bs=1M oflag=seek_bytes seek=${MODEL_OFFSET} conv=sparse status=none
  DEPENDS ${OUTPUT_DIR}/model.img
  VERBATIM
)
add_custom_target(smolvla-base-model-build DEPENDS ${ELF} ${ELF}-ddr.bin ${STEP_ELF} ${OUTPUT_DIR}/model.img)
