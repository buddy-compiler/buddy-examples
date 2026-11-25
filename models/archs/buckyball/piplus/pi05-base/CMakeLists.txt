file(GLOB_RECURSE RECIPE_FRONTEND_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/frontend/*.py")

# What the recipe compiles for this device config; the calibration of the
# model belongs to the recipe.
set(OUTPUT_DIR "${DESIGN_OUTPUT}/models/pi05-base")
# The program is compiled with the device's plugin (DESIGN_DEVICE_OPT), each
# graph with the plugins of every level its target passes through
# (DESIGN_TARGET_OPT_<target>): the vision graph runs on the dense harts and
# their thread, the ranks of the partitioned prefix and denoising step on the
# action and dense harts.
# The boot of every hart of the device: start.S, syscalls.c, device.ld.
set(BOOT_DIR "${DESIGN_REPO}/stack/runtime/boot")
# The operator image of the GP domain of a thread: the operators convert-linalg-to-acc
# leaves in a graph and their entry `operator`, linked by the hart's operators.ld.
set(DENSE_OPERATOR_LINKER "${DESIGN_HART_STACK_dense}/operators.ld")
set(ACTION_OPERATOR_LINKER "${DESIGN_HART_STACK_action}/operators.ld")
list(TRANSFORM DESIGN_DEFS PREPEND "-D" OUTPUT_VARIABLE DESIGN_DEFINES)

# ===----------------------------------------------------------------------===
# Import pi05-base: its graphs, quantized, the device program and the
# program of the first denoising step alone.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/program.mlir
         ${OUTPUT_DIR}/graph-vision.mlir
         ${OUTPUT_DIR}/graph-prefix-action.mlir
         ${OUTPUT_DIR}/graph-prefix-dense.mlir
         ${OUTPUT_DIR}/graph-step-action.mlir
         ${OUTPUT_DIR}/graph-step-dense.mlir
         ${OUTPUT_DIR}/step_program.mlir
         ${OUTPUT_DIR}/vision-arg0.data
         ${OUTPUT_DIR}/vision-arg1.data
         ${OUTPUT_DIR}/prefix-pack-f32.data
         ${OUTPUT_DIR}/prefix-pack-i8.data
         ${OUTPUT_DIR}/step-pack-f32.data
         ${OUTPUT_DIR}/step-pack-i8.data
  COMMAND ${PYTHON} buddy-pytorch-importer.py ${OUTPUT_DIR} ${DESIGN_PB}
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          ${RECIPE_FRONTEND_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${DESIGN_PB}
          ${MODEL_DIR}/pi05-base/model.py
          ${MODEL_DIR}/pi05-base/observation.py
          ${MODEL_DIR}/pi05-base/output/pi05-base/config.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing pi05-base for piplus"
  VERBATIM
)
add_custom_target(pi05-base-import DEPENDS ${OUTPUT_DIR}/program.mlir)

# ===----------------------------------------------------------------------===
# The golden data of the baremetal runners: the inputs of the program (one
# SO100 observation) and the actions of the quantized program in PyTorch;
# the inputs of the first denoising step alone and its velocity.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/actions.data
         ${OUTPUT_DIR}/step_velocity.data
         ${OUTPUT_DIR}/images.data
         ${OUTPUT_DIR}/tokens.data
         ${OUTPUT_DIR}/token_masks.data
         ${OUTPUT_DIR}/noise.data
         ${OUTPUT_DIR}/step_time.data
         ${OUTPUT_DIR}/step_bias.data
         ${OUTPUT_DIR}/step_positions.data
         ${OUTPUT_DIR}/step_cache.data
         ${OUTPUT_DIR}/observation.data
  COMMAND ${PYTHON} buddy-pytorch-golden.py ${TOOLCHAIN}/lib/libnewlib-reference.so ${OUTPUT_DIR}
  DEPENDS ${STACK_ROOT}/compiler/frontend/quant.py
          ${STACK_ROOT}/flake.lock
          ${STACK_ROOT}/scripts/nix/riscv.nix
          ${TOOLCHAIN}/lib/libnewlib-reference.so
          ${CMAKE_CURRENT_SOURCE_DIR}/frontend/graphs.py
          ${CMAKE_CURRENT_SOURCE_DIR}/frontend/program.py
          ${CMAKE_CURRENT_SOURCE_DIR}/frontend/quant/graph.py
          ${CMAKE_CURRENT_SOURCE_DIR}/frontend/quant/reference.py
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-golden.py
          ${MODEL_DIR}/pi05-base/model.py
          ${MODEL_DIR}/pi05-base/observation.py
          ${MODEL_DIR}/pi05-base/output/pi05-base/config.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Computing the pi05-base golden actions for piplus"
  VERBATIM
)
add_custom_target(pi05-base-golden DEPENDS ${OUTPUT_DIR}/actions.data)

# ===----------------------------------------------------------------------===
# The device program of pi05-base: its launches placed on the harts,
# transfers between them, lowered to the device runtime; on the control hart.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT program.o
  BYPRODUCTS ${OUTPUT_DIR}/program_placed.mlir
             ${OUTPUT_DIR}/program_bufferized.mlir
             ${OUTPUT_DIR}/program_lowered.mlir
             ${OUTPUT_DIR}/program_llvm.mlir
             ${OUTPUT_DIR}/program.ll
  COMMAND ${DESIGN_DEVICE_OPT} ${OUTPUT_DIR}/program.mlir -convert-linalg-to-device=design=${DESIGN_PB} -o ${OUTPUT_DIR}/program_placed.mlir
  COMMAND ${DESIGN_DEVICE_OPT} ${OUTPUT_DIR}/program_placed.mlir -convert-elementwise-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs hoist-dynamic-allocs" -canonicalize -cse -o ${OUTPUT_DIR}/program_bufferized.mlir
  COMMAND ${DESIGN_DEVICE_OPT} ${OUTPUT_DIR}/program_bufferized.mlir -lower-device -o ${OUTPUT_DIR}/program_lowered.mlir
  COMMAND ${DESIGN_DEVICE_OPT} ${OUTPUT_DIR}/program_lowered.mlir -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts -o ${OUTPUT_DIR}/program_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/program_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/program.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/program.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -o program.o
  DEPENDS ${OUTPUT_DIR}/program.mlir
          ${DESIGN_DEVICE_PLUGIN}
          ${DESIGN_PB}
  COMMENT "Building the pi05-base program for piplus"
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
  COMMAND ${DESIGN_DEVICE_OPT} ${OUTPUT_DIR}/step_program.mlir -convert-linalg-to-device=design=${DESIGN_PB} -o ${OUTPUT_DIR}/step_program_placed.mlir
  COMMAND ${DESIGN_DEVICE_OPT} ${OUTPUT_DIR}/step_program_placed.mlir -convert-elementwise-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs hoist-dynamic-allocs" -canonicalize -cse -o ${OUTPUT_DIR}/step_program_bufferized.mlir
  COMMAND ${DESIGN_DEVICE_OPT} ${OUTPUT_DIR}/step_program_bufferized.mlir -lower-device -o ${OUTPUT_DIR}/step_program_lowered.mlir
  COMMAND ${DESIGN_DEVICE_OPT} ${OUTPUT_DIR}/step_program_lowered.mlir -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts -o ${OUTPUT_DIR}/step_program_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/step_program_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/step_program.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/step_program.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -o step-program.o
  DEPENDS ${OUTPUT_DIR}/step_program.mlir
          ${DESIGN_DEVICE_PLUGIN}
          ${DESIGN_PB}
  COMMENT "Building the pi05-base step program for piplus"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The vision encoder on the dense harts, a camera a hart.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT vision.o
  BYPRODUCTS ${OUTPUT_DIR}/vision_linalg.mlir
             ${OUTPUT_DIR}/vision_hart.mlir
             ${OUTPUT_DIR}/vision_acc.mlir
             ${OUTPUT_DIR}/vision_host.mlir
             ${OUTPUT_DIR}/vision_banks.mlir
             ${OUTPUT_DIR}/vision_instructions.mlir
             ${OUTPUT_DIR}/vision_addresses.mlir
             ${OUTPUT_DIR}/vision_frontend.mlir
             ${OUTPUT_DIR}/vision_bufferized.mlir
             ${OUTPUT_DIR}/vision_memref.mlir
             ${OUTPUT_DIR}/vision_copies.mlir
             ${OUTPUT_DIR}/vision_llvm.mlir
             ${OUTPUT_DIR}/vision.ll
             ${OUTPUT_DIR}/vision_opt.ll
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/graph-vision.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith,cse))"
            -o ${OUTPUT_DIR}/vision_linalg.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/vision_linalg.mlir "-convert-linalg-to-hart=design=${DESIGN_PB} hart=dense"
            -o ${OUTPUT_DIR}/vision_hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/vision_hart.mlir
            -convert-linalg-to-acc
            -o ${OUTPUT_DIR}/vision_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/vision_acc.mlir "-symbol-privatize=exclude=vision_graph,vision,_binary_operators_image_start"
            -symbol-dce
            -o ${OUTPUT_DIR}/vision_host.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/vision_host.mlir
            -lower-frontend-fake-op
            -o ${OUTPUT_DIR}/vision_frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/vision_frontend.mlir
            -inline
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
            -drop-equivalent-buffer-results=modify-public-functions "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
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
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/vision_instructions.mlir
            -expand-strided-metadata
            -lower-affine
            -canonicalize
            -o ${OUTPUT_DIR}/vision_addresses.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/vision_addresses.mlir "-lower-hart=design=${DESIGN_PB} hart=dense"
            -o ${OUTPUT_DIR}/vision_memref.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/vision_memref.mlir
            -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))"
            -o ${OUTPUT_DIR}/vision_copies.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/vision_copies.mlir
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
            -o ${OUTPUT_DIR}/vision_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/vision_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/vision.ll
  COMMAND ${LLVM_OPT} ${OUTPUT_DIR}/vision.ll -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=default<O2>" -S -o ${OUTPUT_DIR}/vision_opt.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/vision_opt.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -max-store-memcpy=16 -o vision.o
  COMMAND ${ELF_OBJCOPY}
            --redefine-sym _binary_operators_image_start=_binary_vision_operators_image_start vision.o
  DEPENDS ${OUTPUT_DIR}/graph-vision.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${DESIGN_TARGET_PLUGIN_dense}
          ${DESIGN_PB}
  COMMENT "Building the pi05-base vision graph for piplus"
  VERBATIM
)

math(EXPR VISION_OPERATOR_TEXT_BYTES "${DESIGN_PROFILES_dense_THREAD_GPDOMAIN_I_BUF_WORDS} * 4")
add_custom_command(
  OUTPUT vision_operators.o
  BYPRODUCTS ${OUTPUT_DIR}/vision_operators.mlir
             ${OUTPUT_DIR}/vision_operators_llvm.mlir
             ${OUTPUT_DIR}/vision_operators.ll
             ${OUTPUT_DIR}/vision_operators_opt.ll
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/vision_acc.mlir "-symbol-privatize=exclude=operator,gp_space"
            -symbol-dce
            -o ${OUTPUT_DIR}/vision_operators.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/vision_operators.mlir
            -convert-vector-to-scf
            -convert-vector-to-llvm
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-math-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/vision_operators_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/vision_operators_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/vision_operators.ll
  COMMAND ${LLVM_OPT} ${OUTPUT_DIR}/vision_operators.ll -mtriple=riscv64 -mattr=+m,+f,+d,+zve${DESIGN_PROFILES_dense_THREAD_GPDOMAIN_E_LEN}f,+zvl128b "-passes=default<O2>" -S -o ${OUTPUT_DIR}/vision_operators_opt.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/vision_operators_opt.ll -filetype=obj -mtriple=riscv64 -O2 -mattr=+m,+f,+d,+zve${DESIGN_PROFILES_dense_THREAD_GPDOMAIN_E_LEN}f,+zvl128b -riscv-v-vector-bits-min=${DESIGN_PROFILES_dense_THREAD_GPDOMAIN_V_LEN} -riscv-v-vector-bits-max=${DESIGN_PROFILES_dense_THREAD_GPDOMAIN_V_LEN} -target-abi=lp64f -o vision_operators_code.o
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-ld -m elf64lriscv
            --defsym=operator_text_bytes=${VISION_OPERATOR_TEXT_BYTES} -T ${DENSE_OPERATOR_LINKER} vision_operators_code.o -o vision_operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata vision_operators.elf vision_operators.image
  COMMAND ${ELF_OBJCOPY} -I binary -O elf64-littleriscv -B riscv
            --rename-section .data=.rodata,alloc,load,readonly,data,contents
            --set-section-alignment .data=16 vision_operators.image vision_operators.o
  DEPENDS vision.o
          ${DENSE_OPERATOR_LINKER}
          ${LLVM_OPT}
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Building the GP operator image of the pi05-base vision graph"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The graphs of pi05-base on their harts, common subexpressions merged with the conversion to linalg:
# convert-linalg-to-hart gives each row softmax to SoftmaxBall, compiles the
# other FP32 row regions into operators of the GP domain of the thread and puts
# them and the MXINT8 quantization and products on the hart's hardware thread
# as subgraphs launched as early as their inputs allow (<graph>_hart.mlir),
# convert-linalg-to-acc maps MXINT8 products and quantization to bank loops and
# row operators to GP launches; the operators leave the host graph
# (<graph>_host.mlir) for the operator image of the graph, embedded as
# _binary_<graph>_operators_image_start; bank assignment and lower-acc emit
# Ball instructions. lower-hart records their affine loops for the kernel,
# retaining staging memory until completion.
# ===----------------------------------------------------------------------===

# ===----------------------------------------------------------------------===
# The step partitioned across the harts (frontend/partition): its graphs on
# the action hart (the reduce hart and the first rank) and on the dense harts
# (the other ranks), each class of hart with its own image of GP operators.
# The graphs of each class are named here, the symbols the step program
# calls: a change of the partition shows at the link.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT step_action.o
  BYPRODUCTS ${OUTPUT_DIR}/step_action_linalg.mlir
             ${OUTPUT_DIR}/step_action_hart.mlir
             ${OUTPUT_DIR}/step_action_acc.mlir
             ${OUTPUT_DIR}/step_action_host.mlir
             ${OUTPUT_DIR}/step_action_banks.mlir
             ${OUTPUT_DIR}/step_action_instructions.mlir
             ${OUTPUT_DIR}/step_action_addresses.mlir
             ${OUTPUT_DIR}/step_action_frontend.mlir
             ${OUTPUT_DIR}/step_action_bufferized.mlir
             ${OUTPUT_DIR}/step_action_memref.mlir
             ${OUTPUT_DIR}/step_action_copies.mlir
             ${OUTPUT_DIR}/step_action_llvm.mlir
             ${OUTPUT_DIR}/step_action.ll
             ${OUTPUT_DIR}/step_action_opt.ll
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/graph-step-action.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith,cse))"
            -o ${OUTPUT_DIR}/step_action_linalg.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/step_action_linalg.mlir "-convert-linalg-to-hart=design=${DESIGN_PB} hart=action"
            -o ${OUTPUT_DIR}/step_action_hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/step_action_hart.mlir
            -convert-linalg-to-acc
            -o ${OUTPUT_DIR}/step_action_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/step_action_acc.mlir "-symbol-privatize=exclude=step_attention_action,step_attention_action_1,step_mlp_action,step_mlp_action_1,step_mlp_action_2,step_mlp_action_3,step_mlp_action_4,step_reduce_action,step_reduce_action_1,step_reduce_action_2,step_reduce_action_3,step_reduce_action_4,step_reduce_action_5,step_reduce_action_6,_binary_operators_image_start"
            -symbol-dce
            -o ${OUTPUT_DIR}/step_action_host.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/step_action_host.mlir
            -lower-frontend-fake-op
            -o ${OUTPUT_DIR}/step_action_frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/step_action_frontend.mlir
            -inline
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
            -drop-equivalent-buffer-results=modify-public-functions "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -canonicalize
            -o ${OUTPUT_DIR}/step_action_bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/step_action_bufferized.mlir
            -assign-physical-banks
            -o ${OUTPUT_DIR}/step_action_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/step_action_banks.mlir
            -lower-acc
            -o ${OUTPUT_DIR}/step_action_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/step_action_instructions.mlir
            -expand-strided-metadata
            -lower-affine
            -canonicalize
            -o ${OUTPUT_DIR}/step_action_addresses.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/step_action_addresses.mlir "-lower-hart=design=${DESIGN_PB} hart=action"
            -o ${OUTPUT_DIR}/step_action_memref.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/step_action_memref.mlir
            -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))"
            -o ${OUTPUT_DIR}/step_action_copies.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/step_action_copies.mlir
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
            -o ${OUTPUT_DIR}/step_action_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/step_action_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/step_action.ll
  COMMAND ${LLVM_OPT} ${OUTPUT_DIR}/step_action.ll -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=default<O2>" -S -o ${OUTPUT_DIR}/step_action_opt.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/step_action_opt.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -max-store-memcpy=16 -o step_action.o
  COMMAND ${ELF_OBJCOPY}
            --redefine-sym _binary_operators_image_start=_binary_step_action_operators_image_start step_action.o
  DEPENDS ${OUTPUT_DIR}/graph-step-action.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${DESIGN_TARGET_PLUGIN_action}
          ${DESIGN_PB}
  COMMENT "Building the pi05-base step graphs of the action harts for piplus"
  VERBATIM
)

math(EXPR STEP_ACTION_OPERATOR_TEXT_BYTES "${DESIGN_PROFILES_action_THREAD_GPDOMAIN_I_BUF_WORDS} * 4")
add_custom_command(
  OUTPUT step_action_operators.o
  BYPRODUCTS ${OUTPUT_DIR}/step_action_operators.mlir
             ${OUTPUT_DIR}/step_action_operators_llvm.mlir
             ${OUTPUT_DIR}/step_action_operators.ll
             ${OUTPUT_DIR}/step_action_operators_opt.ll
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/step_action_acc.mlir "-symbol-privatize=exclude=operator,gp_space"
            -symbol-dce
            -o ${OUTPUT_DIR}/step_action_operators.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/step_action_operators.mlir
            -convert-vector-to-scf
            -convert-vector-to-llvm
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-math-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/step_action_operators_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/step_action_operators_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/step_action_operators.ll
  COMMAND ${LLVM_OPT} ${OUTPUT_DIR}/step_action_operators.ll -mtriple=riscv64 -mattr=+m,+f,+d,+zve${DESIGN_PROFILES_action_THREAD_GPDOMAIN_E_LEN}f,+zvl128b "-passes=default<O2>" -S -o ${OUTPUT_DIR}/step_action_operators_opt.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/step_action_operators_opt.ll -filetype=obj -mtriple=riscv64 -O2 -mattr=+m,+f,+d,+zve${DESIGN_PROFILES_action_THREAD_GPDOMAIN_E_LEN}f,+zvl128b -riscv-v-vector-bits-min=${DESIGN_PROFILES_action_THREAD_GPDOMAIN_V_LEN} -riscv-v-vector-bits-max=${DESIGN_PROFILES_action_THREAD_GPDOMAIN_V_LEN} -target-abi=lp64f -o step_action_operators_code.o
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-ld -m elf64lriscv
            --defsym=operator_text_bytes=${STEP_ACTION_OPERATOR_TEXT_BYTES} -T ${ACTION_OPERATOR_LINKER} step_action_operators_code.o -o step_action_operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata step_action_operators.elf step_action_operators.image
  COMMAND ${ELF_OBJCOPY} -I binary -O elf64-littleriscv -B riscv
            --rename-section .data=.rodata,alloc,load,readonly,data,contents
            --set-section-alignment .data=16 step_action_operators.image step_action_operators.o
  DEPENDS step_action.o
          ${ACTION_OPERATOR_LINKER}
          ${LLVM_OPT}
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Building the GP operator image of the pi05-base step graphs of the action harts"
  VERBATIM
)

add_custom_command(
  OUTPUT step_dense.o
  BYPRODUCTS ${OUTPUT_DIR}/step_dense_linalg.mlir
             ${OUTPUT_DIR}/step_dense_hart.mlir
             ${OUTPUT_DIR}/step_dense_acc.mlir
             ${OUTPUT_DIR}/step_dense_host.mlir
             ${OUTPUT_DIR}/step_dense_banks.mlir
             ${OUTPUT_DIR}/step_dense_instructions.mlir
             ${OUTPUT_DIR}/step_dense_addresses.mlir
             ${OUTPUT_DIR}/step_dense_frontend.mlir
             ${OUTPUT_DIR}/step_dense_bufferized.mlir
             ${OUTPUT_DIR}/step_dense_memref.mlir
             ${OUTPUT_DIR}/step_dense_copies.mlir
             ${OUTPUT_DIR}/step_dense_llvm.mlir
             ${OUTPUT_DIR}/step_dense.ll
             ${OUTPUT_DIR}/step_dense_opt.ll
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/graph-step-dense.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith,cse))"
            -o ${OUTPUT_DIR}/step_dense_linalg.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/step_dense_linalg.mlir "-convert-linalg-to-hart=design=${DESIGN_PB} hart=dense"
            -o ${OUTPUT_DIR}/step_dense_hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/step_dense_hart.mlir
            -convert-linalg-to-acc
            -o ${OUTPUT_DIR}/step_dense_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/step_dense_acc.mlir "-symbol-privatize=exclude=step_attention_dense,step_attention_dense_1,step_attention_dense_2,step_attention_dense_3,step_mlp_dense,step_mlp_dense_1,_binary_operators_image_start"
            -symbol-dce
            -o ${OUTPUT_DIR}/step_dense_host.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/step_dense_host.mlir
            -lower-frontend-fake-op
            -o ${OUTPUT_DIR}/step_dense_frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/step_dense_frontend.mlir
            -inline
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
            -drop-equivalent-buffer-results=modify-public-functions "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -canonicalize
            -o ${OUTPUT_DIR}/step_dense_bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/step_dense_bufferized.mlir
            -assign-physical-banks
            -o ${OUTPUT_DIR}/step_dense_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/step_dense_banks.mlir
            -lower-acc
            -o ${OUTPUT_DIR}/step_dense_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/step_dense_instructions.mlir
            -expand-strided-metadata
            -lower-affine
            -canonicalize
            -o ${OUTPUT_DIR}/step_dense_addresses.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/step_dense_addresses.mlir "-lower-hart=design=${DESIGN_PB} hart=dense"
            -o ${OUTPUT_DIR}/step_dense_memref.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/step_dense_memref.mlir
            -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))"
            -o ${OUTPUT_DIR}/step_dense_copies.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/step_dense_copies.mlir
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
            -o ${OUTPUT_DIR}/step_dense_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/step_dense_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/step_dense.ll
  COMMAND ${LLVM_OPT} ${OUTPUT_DIR}/step_dense.ll -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=default<O2>" -S -o ${OUTPUT_DIR}/step_dense_opt.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/step_dense_opt.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -max-store-memcpy=16 -o step_dense.o
  COMMAND ${ELF_OBJCOPY}
            --redefine-sym _binary_operators_image_start=_binary_step_dense_operators_image_start step_dense.o
  DEPENDS ${OUTPUT_DIR}/graph-step-dense.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${DESIGN_TARGET_PLUGIN_dense}
          ${DESIGN_PB}
  COMMENT "Building the pi05-base step graphs of the dense harts for piplus"
  VERBATIM
)

math(EXPR STEP_DENSE_OPERATOR_TEXT_BYTES "${DESIGN_PROFILES_dense_THREAD_GPDOMAIN_I_BUF_WORDS} * 4")
add_custom_command(
  OUTPUT step_dense_operators.o
  BYPRODUCTS ${OUTPUT_DIR}/step_dense_operators.mlir
             ${OUTPUT_DIR}/step_dense_operators_llvm.mlir
             ${OUTPUT_DIR}/step_dense_operators.ll
             ${OUTPUT_DIR}/step_dense_operators_opt.ll
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/step_dense_acc.mlir "-symbol-privatize=exclude=operator,gp_space"
            -symbol-dce
            -o ${OUTPUT_DIR}/step_dense_operators.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/step_dense_operators.mlir
            -convert-vector-to-scf
            -convert-vector-to-llvm
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-math-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/step_dense_operators_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/step_dense_operators_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/step_dense_operators.ll
  COMMAND ${LLVM_OPT} ${OUTPUT_DIR}/step_dense_operators.ll -mtriple=riscv64 -mattr=+m,+f,+d,+zve${DESIGN_PROFILES_dense_THREAD_GPDOMAIN_E_LEN}f,+zvl128b "-passes=default<O2>" -S -o ${OUTPUT_DIR}/step_dense_operators_opt.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/step_dense_operators_opt.ll -filetype=obj -mtriple=riscv64 -O2 -mattr=+m,+f,+d,+zve${DESIGN_PROFILES_dense_THREAD_GPDOMAIN_E_LEN}f,+zvl128b -riscv-v-vector-bits-min=${DESIGN_PROFILES_dense_THREAD_GPDOMAIN_V_LEN} -riscv-v-vector-bits-max=${DESIGN_PROFILES_dense_THREAD_GPDOMAIN_V_LEN} -target-abi=lp64f -o step_dense_operators_code.o
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-ld -m elf64lriscv
            --defsym=operator_text_bytes=${STEP_DENSE_OPERATOR_TEXT_BYTES} -T ${DENSE_OPERATOR_LINKER} step_dense_operators_code.o -o step_dense_operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata step_dense_operators.elf step_dense_operators.image
  COMMAND ${ELF_OBJCOPY} -I binary -O elf64-littleriscv -B riscv
            --rename-section .data=.rodata,alloc,load,readonly,data,contents
            --set-section-alignment .data=16 step_dense_operators.image step_dense_operators.o
  DEPENDS step_dense.o
          ${DENSE_OPERATOR_LINKER}
          ${LLVM_OPT}
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Building the GP operator image of the pi05-base step graphs of the dense harts"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The prefix partitioned across the harts (frontend/partition): its graphs on
# the action hart (the reduce hart and the first rank) and on the dense harts
# (the other ranks), each class of hart with its own image of GP operators.
# The graphs of each class are named here, the symbols the program
# calls: a change of the partition shows at the link.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT prefix_action.o
  BYPRODUCTS ${OUTPUT_DIR}/prefix_action_linalg.mlir
             ${OUTPUT_DIR}/prefix_action_hart.mlir
             ${OUTPUT_DIR}/prefix_action_acc.mlir
             ${OUTPUT_DIR}/prefix_action_host.mlir
             ${OUTPUT_DIR}/prefix_action_banks.mlir
             ${OUTPUT_DIR}/prefix_action_instructions.mlir
             ${OUTPUT_DIR}/prefix_action_addresses.mlir
             ${OUTPUT_DIR}/prefix_action_frontend.mlir
             ${OUTPUT_DIR}/prefix_action_bufferized.mlir
             ${OUTPUT_DIR}/prefix_action_memref.mlir
             ${OUTPUT_DIR}/prefix_action_copies.mlir
             ${OUTPUT_DIR}/prefix_action_llvm.mlir
             ${OUTPUT_DIR}/prefix_action.ll
             ${OUTPUT_DIR}/prefix_action_opt.ll
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/graph-prefix-action.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith,cse))"
            -o ${OUTPUT_DIR}/prefix_action_linalg.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/prefix_action_linalg.mlir "-convert-linalg-to-hart=design=${DESIGN_PB} hart=action"
            -o ${OUTPUT_DIR}/prefix_action_hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/prefix_action_hart.mlir
            -convert-linalg-to-acc
            -o ${OUTPUT_DIR}/prefix_action_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/prefix_action_acc.mlir "-symbol-privatize=exclude=prefix_attention_action,prefix_attention_action_1,prefix_mlp_action,prefix_mlp_action_1,prefix_mlp_action_2,prefix_mlp_action_3,prefix_mlp_action_4,prefix_mlp_action_5,prefix_mlp_action_6,prefix_mlp_action_7,prefix_reduce_action,prefix_reduce_action_1,prefix_reduce_action_2,prefix_reduce_action_3,prefix_reduce_action_4,prefix_reduce_action_5,_binary_operators_image_start"
            -symbol-dce
            -o ${OUTPUT_DIR}/prefix_action_host.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/prefix_action_host.mlir
            -lower-frontend-fake-op
            -o ${OUTPUT_DIR}/prefix_action_frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/prefix_action_frontend.mlir
            -inline
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
            -drop-equivalent-buffer-results=modify-public-functions "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -canonicalize
            -o ${OUTPUT_DIR}/prefix_action_bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/prefix_action_bufferized.mlir
            -assign-physical-banks
            -o ${OUTPUT_DIR}/prefix_action_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/prefix_action_banks.mlir
            -lower-acc
            -o ${OUTPUT_DIR}/prefix_action_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/prefix_action_instructions.mlir
            -expand-strided-metadata
            -lower-affine
            -canonicalize
            -o ${OUTPUT_DIR}/prefix_action_addresses.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/prefix_action_addresses.mlir "-lower-hart=design=${DESIGN_PB} hart=action"
            -o ${OUTPUT_DIR}/prefix_action_memref.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/prefix_action_memref.mlir
            -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))"
            -o ${OUTPUT_DIR}/prefix_action_copies.mlir
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/prefix_action_copies.mlir
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
            -o ${OUTPUT_DIR}/prefix_action_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/prefix_action_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/prefix_action.ll
  COMMAND ${LLVM_OPT} ${OUTPUT_DIR}/prefix_action.ll -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=default<O2>" -S -o ${OUTPUT_DIR}/prefix_action_opt.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/prefix_action_opt.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -max-store-memcpy=16 -o prefix_action.o
  COMMAND ${ELF_OBJCOPY}
            --redefine-sym _binary_operators_image_start=_binary_prefix_action_operators_image_start prefix_action.o
  DEPENDS ${OUTPUT_DIR}/graph-prefix-action.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${DESIGN_TARGET_PLUGIN_action}
          ${DESIGN_PB}
  COMMENT "Building the pi05-base prefix graphs of the action harts for piplus"
  VERBATIM
)

math(EXPR PREFIX_ACTION_OPERATOR_TEXT_BYTES "${DESIGN_PROFILES_action_THREAD_GPDOMAIN_I_BUF_WORDS} * 4")
add_custom_command(
  OUTPUT prefix_action_operators.o
  BYPRODUCTS ${OUTPUT_DIR}/prefix_action_operators.mlir
             ${OUTPUT_DIR}/prefix_action_operators_llvm.mlir
             ${OUTPUT_DIR}/prefix_action_operators.ll
             ${OUTPUT_DIR}/prefix_action_operators_opt.ll
  COMMAND ${DESIGN_TARGET_OPT_action} ${OUTPUT_DIR}/prefix_action_acc.mlir "-symbol-privatize=exclude=operator,gp_space"
            -symbol-dce
            -o ${OUTPUT_DIR}/prefix_action_operators.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/prefix_action_operators.mlir
            -convert-vector-to-scf
            -convert-vector-to-llvm
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-math-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/prefix_action_operators_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/prefix_action_operators_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/prefix_action_operators.ll
  COMMAND ${LLVM_OPT} ${OUTPUT_DIR}/prefix_action_operators.ll -mtriple=riscv64 -mattr=+m,+f,+d,+zve${DESIGN_PROFILES_action_THREAD_GPDOMAIN_E_LEN}f,+zvl128b "-passes=default<O2>" -S -o ${OUTPUT_DIR}/prefix_action_operators_opt.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/prefix_action_operators_opt.ll -filetype=obj -mtriple=riscv64 -O2 -mattr=+m,+f,+d,+zve${DESIGN_PROFILES_action_THREAD_GPDOMAIN_E_LEN}f,+zvl128b -riscv-v-vector-bits-min=${DESIGN_PROFILES_action_THREAD_GPDOMAIN_V_LEN} -riscv-v-vector-bits-max=${DESIGN_PROFILES_action_THREAD_GPDOMAIN_V_LEN} -target-abi=lp64f -o prefix_action_operators_code.o
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-ld -m elf64lriscv
            --defsym=operator_text_bytes=${PREFIX_ACTION_OPERATOR_TEXT_BYTES} -T ${ACTION_OPERATOR_LINKER} prefix_action_operators_code.o -o prefix_action_operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata prefix_action_operators.elf prefix_action_operators.image
  COMMAND ${ELF_OBJCOPY} -I binary -O elf64-littleriscv -B riscv
            --rename-section .data=.rodata,alloc,load,readonly,data,contents
            --set-section-alignment .data=16 prefix_action_operators.image prefix_action_operators.o
  DEPENDS prefix_action.o
          ${ACTION_OPERATOR_LINKER}
          ${LLVM_OPT}
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Building the GP operator image of the pi05-base prefix graphs of the action harts"
  VERBATIM
)

add_custom_command(
  OUTPUT prefix_dense.o
  BYPRODUCTS ${OUTPUT_DIR}/prefix_dense_linalg.mlir
             ${OUTPUT_DIR}/prefix_dense_hart.mlir
             ${OUTPUT_DIR}/prefix_dense_acc.mlir
             ${OUTPUT_DIR}/prefix_dense_host.mlir
             ${OUTPUT_DIR}/prefix_dense_banks.mlir
             ${OUTPUT_DIR}/prefix_dense_instructions.mlir
             ${OUTPUT_DIR}/prefix_dense_addresses.mlir
             ${OUTPUT_DIR}/prefix_dense_frontend.mlir
             ${OUTPUT_DIR}/prefix_dense_bufferized.mlir
             ${OUTPUT_DIR}/prefix_dense_memref.mlir
             ${OUTPUT_DIR}/prefix_dense_copies.mlir
             ${OUTPUT_DIR}/prefix_dense_llvm.mlir
             ${OUTPUT_DIR}/prefix_dense.ll
             ${OUTPUT_DIR}/prefix_dense_opt.ll
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/graph-prefix-dense.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith,cse))"
            -o ${OUTPUT_DIR}/prefix_dense_linalg.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/prefix_dense_linalg.mlir "-convert-linalg-to-hart=design=${DESIGN_PB} hart=dense"
            -o ${OUTPUT_DIR}/prefix_dense_hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/prefix_dense_hart.mlir
            -convert-linalg-to-acc
            -o ${OUTPUT_DIR}/prefix_dense_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/prefix_dense_acc.mlir "-symbol-privatize=exclude=prefix_attention_dense,prefix_attention_dense_1,prefix_attention_dense_2,prefix_attention_dense_3,prefix_mlp_dense,prefix_mlp_dense_1,prefix_mlp_dense_2,prefix_mlp_dense_3,prefix_mlp_dense_4,_binary_operators_image_start"
            -symbol-dce
            -o ${OUTPUT_DIR}/prefix_dense_host.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/prefix_dense_host.mlir
            -lower-frontend-fake-op
            -o ${OUTPUT_DIR}/prefix_dense_frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/prefix_dense_frontend.mlir
            -inline
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
            -drop-equivalent-buffer-results=modify-public-functions "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -canonicalize
            -o ${OUTPUT_DIR}/prefix_dense_bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/prefix_dense_bufferized.mlir
            -assign-physical-banks
            -o ${OUTPUT_DIR}/prefix_dense_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/prefix_dense_banks.mlir
            -lower-acc
            -o ${OUTPUT_DIR}/prefix_dense_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/prefix_dense_instructions.mlir
            -expand-strided-metadata
            -lower-affine
            -canonicalize
            -o ${OUTPUT_DIR}/prefix_dense_addresses.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/prefix_dense_addresses.mlir "-lower-hart=design=${DESIGN_PB} hart=dense"
            -o ${OUTPUT_DIR}/prefix_dense_memref.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/prefix_dense_memref.mlir
            -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))"
            -o ${OUTPUT_DIR}/prefix_dense_copies.mlir
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/prefix_dense_copies.mlir
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
            -o ${OUTPUT_DIR}/prefix_dense_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/prefix_dense_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/prefix_dense.ll
  COMMAND ${LLVM_OPT} ${OUTPUT_DIR}/prefix_dense.ll -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=default<O2>" -S -o ${OUTPUT_DIR}/prefix_dense_opt.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/prefix_dense_opt.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -max-store-memcpy=16 -o prefix_dense.o
  COMMAND ${ELF_OBJCOPY}
            --redefine-sym _binary_operators_image_start=_binary_prefix_dense_operators_image_start prefix_dense.o
  DEPENDS ${OUTPUT_DIR}/graph-prefix-dense.mlir
          ${COMPILER_BUILD}/bin/buddy-opt
          ${DESIGN_TARGET_PLUGIN_dense}
          ${DESIGN_PB}
  COMMENT "Building the pi05-base prefix graphs of the dense harts for piplus"
  VERBATIM
)

math(EXPR PREFIX_DENSE_OPERATOR_TEXT_BYTES "${DESIGN_PROFILES_dense_THREAD_GPDOMAIN_I_BUF_WORDS} * 4")
add_custom_command(
  OUTPUT prefix_dense_operators.o
  BYPRODUCTS ${OUTPUT_DIR}/prefix_dense_operators.mlir
             ${OUTPUT_DIR}/prefix_dense_operators_llvm.mlir
             ${OUTPUT_DIR}/prefix_dense_operators.ll
             ${OUTPUT_DIR}/prefix_dense_operators_opt.ll
  COMMAND ${DESIGN_TARGET_OPT_dense} ${OUTPUT_DIR}/prefix_dense_acc.mlir "-symbol-privatize=exclude=operator,gp_space"
            -symbol-dce
            -o ${OUTPUT_DIR}/prefix_dense_operators.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-opt ${OUTPUT_DIR}/prefix_dense_operators.mlir
            -convert-vector-to-scf
            -convert-vector-to-llvm
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-math-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/prefix_dense_operators_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/prefix_dense_operators_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/prefix_dense_operators.ll
  COMMAND ${LLVM_OPT} ${OUTPUT_DIR}/prefix_dense_operators.ll -mtriple=riscv64 -mattr=+m,+f,+d,+zve${DESIGN_PROFILES_dense_THREAD_GPDOMAIN_E_LEN}f,+zvl128b "-passes=default<O2>" -S -o ${OUTPUT_DIR}/prefix_dense_operators_opt.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/prefix_dense_operators_opt.ll -filetype=obj -mtriple=riscv64 -O2 -mattr=+m,+f,+d,+zve${DESIGN_PROFILES_dense_THREAD_GPDOMAIN_E_LEN}f,+zvl128b -riscv-v-vector-bits-min=${DESIGN_PROFILES_dense_THREAD_GPDOMAIN_V_LEN} -riscv-v-vector-bits-max=${DESIGN_PROFILES_dense_THREAD_GPDOMAIN_V_LEN} -target-abi=lp64f -o prefix_dense_operators_code.o
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-ld -m elf64lriscv
            --defsym=operator_text_bytes=${PREFIX_DENSE_OPERATOR_TEXT_BYTES} -T ${DENSE_OPERATOR_LINKER} prefix_dense_operators_code.o -o prefix_dense_operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata prefix_dense_operators.elf prefix_dense_operators.image
  COMMAND ${ELF_OBJCOPY} -I binary -O elf64-littleriscv -B riscv
            --rename-section .data=.rodata,alloc,load,readonly,data,contents
            --set-section-alignment .data=16 prefix_dense_operators.image prefix_dense_operators.o
  DEPENDS prefix_dense.o
          ${DENSE_OPERATOR_LINKER}
          ${LLVM_OPT}
          ${COMPILER_BUILD}/bin/buddy-opt
          ${COMPILER_BUILD}/bin/buddy-translate
          ${COMPILER_BUILD}/bin/buddy-llc
  COMMENT "Building the GP operator image of the pi05-base prefix graphs of the dense harts"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The baremetal image of pi05-base: the runner (every hart enters it),
# the program and the graphs, the device runtime, the bindings of the harts
# and the loaders of their threads (a hart with threads starts its kernel for
# every job); the parameters, inputs and reference actions embedded. It goes with
# the workloads of the device, where BEMU finds it.
# ===----------------------------------------------------------------------===
math(EXPR MODEL_OFFSET "${DESIGN_DEVICE_DDR_BYTES} - ${DESIGN_DEVICE_DDR_MODEL}")
math(EXPR MODEL_IMAGE "${DESIGN_DEVICE_DDR_BASE} + ${MODEL_OFFSET}" OUTPUT_FORMAT HEXADECIMAL)
file(READ ${MODEL_DIR}/pi05-base/output/pi05-base/config.json MODEL_CONFIG)
string(JSON CHUNK_SIZE GET "${MODEL_CONFIG}" chunk_size)
string(JSON ACTION_DIM GET "${MODEL_CONFIG}" max_action_dim)
math(EXPR ACTION_ELEMENTS "${CHUNK_SIZE} * ${ACTION_DIM}")
set(ELF "${DESIGN_OUTPUT}/workloads/models/piplus-pi05-base-baremetal")
add_custom_command(
  OUTPUT ${ELF}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DESIGN_OUTPUT}/workloads/models"
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++17 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -fno-exceptions -fno-rtti ${DESIGN_DEFINES} -DDEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART} -DMODEL_IMAGE=${MODEL_IMAGE}ULL -DACTION_ELEMENTS=${ACTION_ELEMENTS} -I${LIB_DIR} -I${DESIGN_DEVICE_STACK}/runtime -I${DESIGN_REPO}/stack/runtime/program -c ${CMAKE_CURRENT_SOURCE_DIR}/baremetal.cpp -o baremetal.o
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany -specs=nano.specs -specs=nosys.specs -nostartfiles -Wl,-T,${BOOT_DIR}/device.ld ${DESIGN_DEFINES} -DWORKLOAD_REPORT_HART=${DESIGN_WORKLOAD_REPORT_HART} -DDEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART} -I${DESIGN_DEVICE_STACK}/runtime -I${DESIGN_REPO}/stack/runtime/program ${BOOT_DIR}/start.S ${BOOT_DIR}/syscalls.c baremetal.o program.o vision.o prefix_action.o prefix_dense.o step_action.o step_dense.o vision_operators.o prefix_action_operators.o prefix_dense_operators.o step_action_operators.o step_dense_operators.o ${DESIGN_HART_BUILD_control}/execute.o ${DESIGN_HART_BUILD_dense}/execute.o ${DESIGN_HART_BUILD_action}/execute.o ${DESIGN_THREAD_BUILD_dense}/loader.o ${DESIGN_THREAD_BUILD_action}/loader.o $<TARGET_FILE:piplus-runtime> $<TARGET_FILE:runtime-program> $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:runtime> $<TARGET_FILE:kernel> -lm -lgcc -o ${ELF}
  DEPENDS ${BOOT_DIR}/start.S
          ${BOOT_DIR}/syscalls.c
          ${BOOT_DIR}/device.ld
          ${CMAKE_CURRENT_SOURCE_DIR}/baremetal.cpp
          model.h
          ${LIB_DIR}/ModelImage.h
          ${LIB_DIR}/testutils.h
          program.o
          vision.o
          prefix_action.o
          prefix_dense.o
          step_action.o
          step_dense.o
          vision_operators.o
          prefix_action_operators.o
          prefix_dense_operators.o
          step_action_operators.o
          step_dense_operators.o
          ${OUTPUT_DIR}/actions.data
          ${DESIGN_DEVICE_STACK}/runtime/device.h
          piplus-runtime
          runtime-program
          runtime-baremetal
          runtime
          kernel
  COMMENT "Linking pi05-base for the piplus device"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The baremetal image of the first denoising step of pi05-base alone: the
# runner (baremetal-step.cpp), the step program and the step graphs of the
# action and dense harts (partitioned), as the image of the policy; the parameters of the step, its inputs (from the quantized
# program on the observation) and its reference velocity embedded.
# ===----------------------------------------------------------------------===
set(STEP_ELF "${DESIGN_OUTPUT}/workloads/models/piplus-pi05-base-step-baremetal")
add_custom_command(
  OUTPUT ${STEP_ELF}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DESIGN_OUTPUT}/workloads/models"
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++20 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -fno-exceptions -fno-rtti ${DESIGN_DEFINES} -DDEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART} -I${LIB_DIR} -I${DESIGN_DEVICE_STACK}/runtime -I${DESIGN_REPO}/stack/runtime/program
            --embed-dir=${OUTPUT_DIR} -c ${CMAKE_CURRENT_SOURCE_DIR}/baremetal-step.cpp -o baremetal-step.o
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany -specs=nano.specs -specs=nosys.specs -nostartfiles -Wl,-T,${BOOT_DIR}/device.ld ${DESIGN_DEFINES} -DWORKLOAD_REPORT_HART=${DESIGN_WORKLOAD_REPORT_HART} -DDEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART} -I${DESIGN_DEVICE_STACK}/runtime -I${DESIGN_REPO}/stack/runtime/program
            --embed-dir=${OUTPUT_DIR} ${BOOT_DIR}/start.S ${BOOT_DIR}/syscalls.c baremetal-step.o step-program.o step_action.o step_dense.o step_action_operators.o step_dense_operators.o ${DESIGN_HART_BUILD_control}/execute.o ${DESIGN_HART_BUILD_dense}/execute.o ${DESIGN_HART_BUILD_action}/execute.o ${DESIGN_THREAD_BUILD_dense}/loader.o ${DESIGN_THREAD_BUILD_action}/loader.o $<TARGET_FILE:piplus-runtime> $<TARGET_FILE:runtime-program> $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:runtime> $<TARGET_FILE:kernel> -lm -lgcc -o ${STEP_ELF}
  DEPENDS ${BOOT_DIR}/start.S
          ${BOOT_DIR}/syscalls.c
          ${BOOT_DIR}/device.ld
          ${CMAKE_CURRENT_SOURCE_DIR}/baremetal-step.cpp
          ${LIB_DIR}/testutils.h
          step-program.o
          step_action.o
          step_dense.o
          step_action_operators.o
          step_dense_operators.o
          ${OUTPUT_DIR}/step-pack-f32.data
          ${OUTPUT_DIR}/step-pack-i8.data
          ${OUTPUT_DIR}/noise.data
          ${OUTPUT_DIR}/step_time.data
          ${OUTPUT_DIR}/step_bias.data
          ${OUTPUT_DIR}/step_positions.data
          ${OUTPUT_DIR}/step_cache.data
          ${OUTPUT_DIR}/step_velocity.data
          ${DESIGN_DEVICE_STACK}/runtime/device.h
          piplus-runtime
          runtime-program
          runtime-baremetal
          runtime
          kernel
  COMMENT "Linking the first pi05-base step for the piplus device"
  VERBATIM
)

add_library(pi05-base-buckyball-piplus-model SHARED EXCLUDE_FROM_ALL entry.cpp program.o vision.o prefix_action.o
  prefix_dense.o step_action.o step_dense.o vision_operators.o prefix_action_operators.o
  prefix_dense_operators.o step_action_operators.o step_dense_operators.o
)
target_link_libraries(pi05-base-buckyball-piplus-model
  PRIVATE piplus-runtime-linux
)
target_link_options(pi05-base-buckyball-piplus-model PRIVATE "LINKER:--no-undefined")
set_target_properties(pi05-base-buckyball-piplus-model
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN;/lib"
)
get_filename_component(FLATBUFFERS_HEADER "${TOOLCHAIN}/include/flatbuffers/flatbuffers.h" REALPATH)
get_filename_component(FLATBUFFERS_INCLUDE "${FLATBUFFERS_HEADER}/../.." ABSOLUTE)
add_library(pi05-base-buckyball-piplus-runner SHARED EXCLUDE_FROM_ALL runner.cpp)
target_include_directories(pi05-base-buckyball-piplus-runner
  PRIVATE ${LIB_DIR}
          ${RUNTIME_DIR}/services
          ${FLATBUFFERS_INCLUDE}
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/runtime/include
          ${DESIGN_COMPILER}/runtime/include
)
target_compile_definitions(pi05-base-buckyball-piplus-runner
  PRIVATE ACTION_ELEMENTS=${ACTION_ELEMENTS}
)
target_link_libraries(pi05-base-buckyball-piplus-runner
  PRIVATE dl
)
set_target_properties(pi05-base-buckyball-piplus-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN;/lib"
)
configure_file(pi05-base.rhal.mlir.in ${OUTPUT_DIR}/pi05-base.rhal.mlir @ONLY)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/pi05-base.rax
  COMMAND ${COMPILER_BUILD}/bin/rax-pack ${OUTPUT_DIR}/pi05-base.rhal.mlir
          --embed-payload
          -o ${OUTPUT_DIR}/pi05-base.rax
  DEPENDS ${OUTPUT_DIR}/pi05-base.rhal.mlir
          ${COMPILER_BUILD}/bin/rax-pack
          pi05-base-buckyball-piplus-model
          pi05-base-buckyball-piplus-runner
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)
# Linux runs the RAX entry on the observation of the baremetal program and
# reports it against the same reference.
file(WRITE ${OUTPUT_DIR}/run.args
  "--model\npi05-base.rax\n--input\nobservation.data\n--reference\nactions.data\n--output\nactions.f32\n"
)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/model.img
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}/ddr
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${OUTPUT_DIR}/vision-arg0.data ${OUTPUT_DIR}/ddr/vision-arg0.data
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${OUTPUT_DIR}/vision-arg1.data ${OUTPUT_DIR}/ddr/vision-arg1.data
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${OUTPUT_DIR}/prefix-pack-f32.data ${OUTPUT_DIR}/ddr/prefix-pack-f32.data
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${OUTPUT_DIR}/prefix-pack-i8.data ${OUTPUT_DIR}/ddr/prefix-pack-i8.data
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${OUTPUT_DIR}/step-pack-f32.data ${OUTPUT_DIR}/ddr/step-pack-f32.data
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${OUTPUT_DIR}/step-pack-i8.data ${OUTPUT_DIR}/ddr/step-pack-i8.data
  COMMAND ${CMAKE_BINARY_DIR}/model-pack ${OUTPUT_DIR}/model.img ${OUTPUT_DIR} pi05-base.rax
            --mapped run.args observation.data actions.data
            --mapped ddr/vision-arg0.data
            --mapped ddr/vision-arg1.data
            --mapped ddr/prefix-pack-f32.data
            --mapped ddr/prefix-pack-i8.data
            --mapped ddr/step-pack-f32.data
            --mapped ddr/step-pack-i8.data
  DEPENDS ${OUTPUT_DIR}/pi05-base.rax
          ${OUTPUT_DIR}/run.args
          ${OUTPUT_DIR}/observation.data
          ${OUTPUT_DIR}/actions.data
          ${OUTPUT_DIR}/vision-arg0.data
          ${OUTPUT_DIR}/vision-arg1.data
          ${OUTPUT_DIR}/prefix-pack-f32.data
          ${OUTPUT_DIR}/prefix-pack-i8.data
          ${OUTPUT_DIR}/step-pack-f32.data
          ${OUTPUT_DIR}/step-pack-i8.data
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
add_custom_target(pi05-base-model-build DEPENDS ${ELF} ${ELF}-ddr.bin ${STEP_ELF} ${OUTPUT_DIR}/model.img)
# The GVN slices of this model, which `bbdev gvn-bemu --debug` builds.
add_subdirectory(gvn)
