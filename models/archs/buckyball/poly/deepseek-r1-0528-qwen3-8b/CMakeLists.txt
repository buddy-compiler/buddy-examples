file(GLOB_RECURSE RECIPE_FRONTEND_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/frontend/*.py")

# What the recipe compiles for this device config; the calibration of the
# model belongs to the recipe.
set(OUTPUT_DIR "${DESIGN_OUTPUT}/models/deepseek-r1-0528-qwen3-8b")
set(CALIBRATION "${CMAKE_CURRENT_SOURCE_DIR}/output/calibration.json")
set(WEIGHTS "${MODEL_DIR}/deepseek-r1-0528-qwen3-8b/output/deepseek-r1-0528-qwen3-8b")
# The program is compiled with the device's plugin (DESIGN_DEVICE_OPT); the
# graphs of the poly harts with the plugins of the device and the poly hart
# (DESIGN_HART_OPT_poly), their attention and ffn thread work with those of
# its target (DESIGN_TARGET_OPT_attention, _ffn); the graphs of the reduce
# hart, which has no hardware thread, with those of its target.
set(REDUCE_HART ${DESIGN_DEVICE_HARTS_${DESIGN_WORKLOAD_CONTROL_HART}_COMPONENT})
set(REDUCE_TARGET ${DESIGN_DEVICE_HARTS_${DESIGN_WORKLOAD_CONTROL_HART}_PROFILE})
set(OPERATOR_LINKER "${DESIGN_HART_STACK_poly}/operators.ld")
math(EXPR OPERATOR_TEXT_BYTES "${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_I_BUF_WORDS} * 4")
# The boot of every hart of the device: start.S, syscalls.c, device.ld.
set(BOOT_DIR "${DESIGN_REPO}/stack/runtime/boot")
list(TRANSFORM DESIGN_DEFS PREPEND "-D" OUTPUT_VARIABLE DESIGN_DEFINES)

# ===----------------------------------------------------------------------===
# Import DeepSeek-R1-0528-Qwen3-8B for any number of tokens: the device program and the
# graphs of each class of hart.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/program.mlir
         ${OUTPUT_DIR}/vocab.txt
         ${OUTPUT_DIR}/input-order.txt
         ${OUTPUT_DIR}/candidates.txt
         ${OUTPUT_DIR}/graph-reduce.mlir
         ${OUTPUT_DIR}/graph-poly.mlir
         ${OUTPUT_DIR}/prompt/pack-f32.data
         ${OUTPUT_DIR}/prompt/pack-i8.data
         ${OUTPUT_DIR}/prompt/input-order.txt
  BYPRODUCTS ${OUTPUT_DIR}/01-import.txt
             ${OUTPUT_DIR}/02-permuted.txt
             ${OUTPUT_DIR}/03-quanted.txt
             ${OUTPUT_DIR}/04-partitioned.txt
  COMMAND ${PYTHON} buddy-pytorch-importer.py ${DESIGN_PB} ${OUTPUT_DIR}
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          ${RECIPE_FRONTEND_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/../topk.py
          ${CMAKE_CURRENT_SOURCE_DIR}/../cache.py
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${DESIGN_PB}
          ${MODEL_DIR}/deepseek-r1-0528-qwen3-8b/model.py
          ${WEIGHTS}/config.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing the DeepSeek-R1-0528-Qwen3-8B graph"
  VERBATIM
)
add_custom_target(deepseek-r1-0528-qwen3-8b-import DEPENDS ${OUTPUT_DIR}/program.mlir)

# ===----------------------------------------------------------------------===
# The golden data of the baremetal runner: for the prompt of input.txt, the
# inputs of the program in its order, its outputs in PyTorch for the prompt
# and the decode steps after it, and their sizes.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/prompt/inputs.data
         ${OUTPUT_DIR}/prompt/sizes.txt
         ${OUTPUT_DIR}/prompt/ids.data
  COMMAND ${PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/../prepare-inputs.py deepseek-r1-0528-qwen3-8b ${OUTPUT_DIR}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../prepare-inputs.py
          ${CMAKE_CURRENT_SOURCE_DIR}/frontend/partition/graph.py
          ${STACK_ROOT}/scripts/design.py
          ${DESIGN_PB}
          ${OUTPUT_DIR}/input-order.txt
          ${OUTPUT_DIR}/candidates.txt
          ${MODEL_DIR}/deepseek-r1-0528-qwen3-8b/model.py
          ${MODEL_DIR}/deepseek-r1-0528-qwen3-8b/input.txt
          ${WEIGHTS}/config.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Preparing deepseek-r1-0528-qwen3-8b workload inputs"
  VERBATIM
)

add_custom_command(
  OUTPUT ${OUTPUT_DIR}/prompt/outputs.data
         ${OUTPUT_DIR}/prompt/candidates.data
         ${OUTPUT_DIR}/prompt/tokens.data
  COMMAND ${PYTHON} buddy-pytorch-golden.py ${DESIGN_PB} ${OUTPUT_DIR}
  DEPENDS ${OUTPUT_DIR}/prompt/inputs.data
          ${OUTPUT_DIR}/prompt/sizes.txt
          ${STACK_ROOT}/compiler/frontend/quant.py
          ${STACK_ROOT}/scripts/design.py
          ${CMAKE_CURRENT_SOURCE_DIR}/frontend/reference.py
          ${CMAKE_CURRENT_SOURCE_DIR}/frontend/partition/graph.py
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-golden.py
          ${DESIGN_PB}
          ${OUTPUT_DIR}/input-order.txt
          ${MODEL_DIR}/deepseek-r1-0528-qwen3-8b/model.py
          ${MODEL_DIR}/deepseek-r1-0528-qwen3-8b/input.txt
          ${WEIGHTS}/config.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Computing the DeepSeek-R1-0528-Qwen3-8B golden outputs of the prompt"
  VERBATIM
)
add_custom_target(deepseek-r1-0528-qwen3-8b-golden DEPENDS ${OUTPUT_DIR}/prompt/outputs.data)

# ===----------------------------------------------------------------------===
# The device program of DeepSeek-R1-0528-Qwen3-8B placed on the harts and lowered to the
# device runtime (program_placed.mlir, program_bufferized.mlir,
# program_lowered.mlir).
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/program_lowered.mlir
  BYPRODUCTS ${OUTPUT_DIR}/program_placed.mlir
             ${OUTPUT_DIR}/program_bufferized.mlir
  COMMAND ${DESIGN_DEVICE_OPT} ${OUTPUT_DIR}/program.mlir -convert-linalg-to-device=design=${DESIGN_PB} -o ${OUTPUT_DIR}/program_placed.mlir
  COMMAND ${DESIGN_DEVICE_OPT} ${OUTPUT_DIR}/program_placed.mlir -convert-elementwise-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs hoist-dynamic-allocs" -canonicalize -cse -o ${OUTPUT_DIR}/program_bufferized.mlir
  COMMAND ${DESIGN_DEVICE_OPT} ${OUTPUT_DIR}/program_bufferized.mlir -lower-device -o ${OUTPUT_DIR}/program_lowered.mlir
  DEPENDS ${OUTPUT_DIR}/program.mlir
          ${DESIGN_DEVICE_PLUGIN}
          ${DESIGN_PB}
  COMMENT "Placing and lowering the DeepSeek-R1-0528-Qwen3-8B program on poly"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The graphs of DeepSeek-R1-0528-Qwen3-8B on the poly harts through the hart level, common
# subexpressions merged with the conversion to linalg:
# convert-linalg-to-hart puts their MXINT8 quantizations and products on the
# attention or ffn threads as subgraphs launched as early as their inputs
# allow (poly_hart.mlir),
# lower-frontend-fake-op computes the rest of the frontend operations
# (poly_frontend.mlir), each segment's graph inlined into it is bufferized
# (poly_bufferized.mlir) and lower-hart makes the transfers DDR buffers
# and the launches batches of asynchronous kernel jobs
# (poly_memref.mlir). Each thread profile gets its own acc and bank lowering.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/poly_memref.mlir
  BYPRODUCTS ${OUTPUT_DIR}/poly_linalg.mlir
             ${OUTPUT_DIR}/poly_hart.mlir
             ${OUTPUT_DIR}/poly_attention_acc.mlir
             ${OUTPUT_DIR}/poly_ffn_acc.mlir
             ${OUTPUT_DIR}/poly_host.mlir
             ${OUTPUT_DIR}/poly_attention_banks.mlir
             ${OUTPUT_DIR}/poly_attention_instructions.mlir
             ${OUTPUT_DIR}/poly_ffn_banks.mlir
             ${OUTPUT_DIR}/poly_ffn_instructions.mlir
             ${OUTPUT_DIR}/poly_addresses.mlir
             ${OUTPUT_DIR}/poly_frontend.mlir
             ${OUTPUT_DIR}/poly_inlined.mlir
             ${OUTPUT_DIR}/poly_forwarded.mlir
             ${OUTPUT_DIR}/poly_bufferized.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUTPUT_DIR}/graph-poly.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith,canonicalize,cse))" -o ${OUTPUT_DIR}/poly_linalg.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUTPUT_DIR}/poly_linalg.mlir -inline "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUTPUT_DIR}/poly_hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUTPUT_DIR}/poly_hart.mlir
            -canonicalize
            -convert-linalg-to-acc
            -o ${OUTPUT_DIR}/poly_attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUTPUT_DIR}/poly_attention_acc.mlir
            -convert-linalg-to-acc
            -o ${OUTPUT_DIR}/poly_ffn_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUTPUT_DIR}/poly_ffn_acc.mlir -prune-gp-operators -symbol-dce -o ${OUTPUT_DIR}/poly_host.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUTPUT_DIR}/poly_host.mlir -lower-frontend-fake-op -o ${OUTPUT_DIR}/poly_frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUTPUT_DIR}/poly_frontend.mlir -inline -eliminate-empty-tensors -o ${OUTPUT_DIR}/poly_inlined.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUTPUT_DIR}/poly_inlined.mlir -forward-materialized-tensors -o ${OUTPUT_DIR}/poly_forwarded.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUTPUT_DIR}/poly_forwarded.mlir -convert-elementwise-to-linalg -linalg-fuse-elementwise-ops -resolve-shaped-type-result-dims -canonicalize "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map" -drop-equivalent-buffer-results=modify-public-functions "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs" -ownership-based-buffer-deallocation -buffer-deallocation-simplification -bufferization-lower-deallocations -canonicalize -o ${OUTPUT_DIR}/poly_bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUTPUT_DIR}/poly_bufferized.mlir
            -assign-physical-banks
            -o ${OUTPUT_DIR}/poly_attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUTPUT_DIR}/poly_attention_banks.mlir
            -lower-acc
            -o ${OUTPUT_DIR}/poly_attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUTPUT_DIR}/poly_attention_instructions.mlir
            -assign-physical-banks
            -o ${OUTPUT_DIR}/poly_ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUTPUT_DIR}/poly_ffn_banks.mlir
            -lower-acc
            -o ${OUTPUT_DIR}/poly_ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUTPUT_DIR}/poly_ffn_instructions.mlir -expand-strided-metadata -lower-affine -canonicalize -o ${OUTPUT_DIR}/poly_addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUTPUT_DIR}/poly_addresses.mlir "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUTPUT_DIR}/poly_memref.mlir
  DEPENDS ${OUTPUT_DIR}/graph-poly.mlir
          ${DESIGN_HART_PLUGIN_poly}
          ${DESIGN_PB}
  COMMENT "Lowering the DeepSeek-R1-0528-Qwen3-8B graphs to the poly harts"
  VERBATIM
)
add_custom_target(deepseek-r1-0528-qwen3-8b-hart DEPENDS ${OUTPUT_DIR}/poly_memref.mlir)

# ===----------------------------------------------------------------------===
# The program of DeepSeek-R1-0528-Qwen3-8B to RISC-V on the reduce hart (program.ll).
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT program.o
  BYPRODUCTS ${OUTPUT_DIR}/program_llvm.mlir
             ${OUTPUT_DIR}/program.ll
             ${OUTPUT_DIR}/program_opt.ll
  COMMAND ${DESIGN_DEVICE_OPT} ${OUTPUT_DIR}/program_lowered.mlir -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts -o ${OUTPUT_DIR}/program_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/program_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/program.ll
  COMMAND ${LLVM_OPT} ${OUTPUT_DIR}/program.ll -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=default<O2>" -S -o ${OUTPUT_DIR}/program_opt.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/program_opt.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -o program.o
  DEPENDS ${OUTPUT_DIR}/program_lowered.mlir
          ${DESIGN_DEVICE_PLUGIN}
  COMMENT "Building the DeepSeek-R1-0528-Qwen3-8B program for poly"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The reduce hart runs ordered sum chains on its accelerator;
# normalization and quantization stay on its CPU.
# lower-frontend-fake-op computes the quantization (reduce_frontend.mlir),
# each graph inlined into its function writes its outputs into the buffers
# of the program (reduce_bufferized.mlir).
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT reduce.o
  BYPRODUCTS ${OUTPUT_DIR}/reduce_linalg.mlir
             ${OUTPUT_DIR}/reduce_quantized.mlir
             ${OUTPUT_DIR}/reduce_frontend.mlir
             ${OUTPUT_DIR}/reduce_inlined.mlir
             ${OUTPUT_DIR}/reduce_forwarded.mlir
             ${OUTPUT_DIR}/reduce_elementwise.mlir
             ${OUTPUT_DIR}/reduce_hart.mlir
             ${OUTPUT_DIR}/reduce_acc.mlir
             ${OUTPUT_DIR}/reduce_host.mlir
             ${OUTPUT_DIR}/reduce_bufferized.mlir
             ${OUTPUT_DIR}/reduce_banks.mlir
             ${OUTPUT_DIR}/reduce_instructions.mlir
             ${OUTPUT_DIR}/reduce_copies.mlir
             ${OUTPUT_DIR}/reduce_llvm.mlir
             ${OUTPUT_DIR}/reduce.ll
             ${OUTPUT_DIR}/reduce_opt.ll
  COMMAND ${DESIGN_TARGET_OPT_${REDUCE_TARGET}} ${OUTPUT_DIR}/graph-reduce.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith,canonicalize,cse))"
            -o ${OUTPUT_DIR}/reduce_linalg.mlir
  COMMAND ${DESIGN_TARGET_OPT_${REDUCE_TARGET}} ${OUTPUT_DIR}/reduce_linalg.mlir
            -inline "-convert-linalg-to-hart=design=${DESIGN_PB} hart=${REDUCE_HART}"
            -convert-linalg-to-acc
            -o ${OUTPUT_DIR}/reduce_quantized.mlir
  COMMAND ${DESIGN_TARGET_OPT_${REDUCE_TARGET}} ${OUTPUT_DIR}/reduce_quantized.mlir
            -lower-frontend-fake-op
            -o ${OUTPUT_DIR}/reduce_frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_${REDUCE_TARGET}} ${OUTPUT_DIR}/reduce_frontend.mlir
            -inline
            -eliminate-empty-tensors
            -o ${OUTPUT_DIR}/reduce_inlined.mlir
  COMMAND ${DESIGN_TARGET_OPT_${REDUCE_TARGET}} ${OUTPUT_DIR}/reduce_inlined.mlir
            -forward-materialized-tensors
            -o ${OUTPUT_DIR}/reduce_forwarded.mlir
  COMMAND ${DESIGN_TARGET_OPT_${REDUCE_TARGET}} ${OUTPUT_DIR}/reduce_forwarded.mlir
            -convert-elementwise-to-linalg
            -o ${OUTPUT_DIR}/reduce_elementwise.mlir
  COMMAND ${DESIGN_TARGET_OPT_${REDUCE_TARGET}} ${OUTPUT_DIR}/reduce_elementwise.mlir "-convert-linalg-to-hart=design=${DESIGN_PB} hart=${REDUCE_HART}"
            -o ${OUTPUT_DIR}/reduce_hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_${REDUCE_TARGET}} ${OUTPUT_DIR}/reduce_hart.mlir
            -convert-linalg-to-acc
            -o ${OUTPUT_DIR}/reduce_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_${REDUCE_TARGET}} ${OUTPUT_DIR}/reduce_acc.mlir
            -prune-gp-operators
            -symbol-dce
            -o ${OUTPUT_DIR}/reduce_host.mlir
  COMMAND ${DESIGN_TARGET_OPT_${REDUCE_TARGET}} ${OUTPUT_DIR}/reduce_host.mlir
            -canonicalize "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
            -drop-equivalent-buffer-results=modify-public-functions "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -canonicalize
            -o ${OUTPUT_DIR}/reduce_bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_${REDUCE_TARGET}} ${OUTPUT_DIR}/reduce_bufferized.mlir
            -assign-physical-banks
            -o ${OUTPUT_DIR}/reduce_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_${REDUCE_TARGET}} ${OUTPUT_DIR}/reduce_banks.mlir
            -lower-acc
            -o ${OUTPUT_DIR}/reduce_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_${REDUCE_TARGET}} ${OUTPUT_DIR}/reduce_instructions.mlir
            -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))"
            -o ${OUTPUT_DIR}/reduce_copies.mlir
  COMMAND ${DESIGN_TARGET_OPT_${REDUCE_TARGET}} ${OUTPUT_DIR}/reduce_copies.mlir
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
            -o ${OUTPUT_DIR}/reduce_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/reduce_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/reduce.ll
  COMMAND ${LLVM_OPT} ${OUTPUT_DIR}/reduce.ll -mtriple=riscv64 -mattr=+xbuckyball,+m,+a,+f,+d,+c "-passes=default<O2>" -S -o ${OUTPUT_DIR}/reduce_opt.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/reduce_opt.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -mattr=+xbuckyball,+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -o reduce.o
  DEPENDS ${OUTPUT_DIR}/graph-reduce.mlir
          ${DESIGN_TARGET_PLUGIN_${REDUCE_TARGET}}
  COMMENT "Building the deepseek-r1-0528-qwen3-8b graphs of the reduce hart for poly"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The graphs of DeepSeek-R1-0528-Qwen3-8B on the poly harts from the hart level
# (poly_memref.mlir) to RISC-V: CPU control code records the Ball instruction
# streams that the hart's hardware threads execute.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT poly.o
  BYPRODUCTS ${OUTPUT_DIR}/poly_copies.mlir
             ${OUTPUT_DIR}/poly_llvm.mlir
             ${OUTPUT_DIR}/poly.ll
             ${OUTPUT_DIR}/poly_opt.ll
  COMMAND ${DESIGN_HART_OPT_poly} ${OUTPUT_DIR}/poly_memref.mlir -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUTPUT_DIR}/poly_copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUTPUT_DIR}/poly_copies.mlir -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm -reconcile-unrealized-casts -o ${OUTPUT_DIR}/poly_llvm.mlir
  COMMAND ${COMPILER_BUILD}/bin/buddy-translate ${OUTPUT_DIR}/poly_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/poly.ll
  COMMAND ${LLVM_OPT} ${OUTPUT_DIR}/poly.ll -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=default<O2>" -S -o ${OUTPUT_DIR}/poly_opt.ll
  COMMAND ${COMPILER_BUILD}/bin/buddy-llc ${OUTPUT_DIR}/poly_opt.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -o poly.o
  DEPENDS ${OUTPUT_DIR}/poly_memref.mlir
          ${DESIGN_HART_PLUGIN_poly}
  COMMENT "Building the DeepSeek-R1-0528-Qwen3-8B graphs of the poly harts for poly"
  VERBATIM
)

# The GP operators of the poly graphs, the image the threads of the poly
# harts load (_binary_operators_image_start).
add_custom_command(
  OUTPUT operators.embed.o
  BYPRODUCTS ${OUTPUT_DIR}/operators.mlir
             ${OUTPUT_DIR}/operators_llvm.mlir
             ${OUTPUT_DIR}/operators.ll
             ${OUTPUT_DIR}/operators_opt.ll
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUTPUT_DIR}/poly_ffn_acc.mlir "-symbol-privatize=exclude=operator,gp_space"
            -symbol-dce
            -verify-gp-target
            -o ${OUTPUT_DIR}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${OUTPUT_DIR}/operators.mlir
            -convert-vector-to-scf
            -convert-vector-to-llvm
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-math-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm=use-aligned-alloc
            -convert-func-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/operators_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/operators_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/operators.ll
  COMMAND ${LLVM_OPT} ${OUTPUT_DIR}/operators.ll -mtriple=riscv64 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR} "-passes=default<O2>" -S -o ${OUTPUT_DIR}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/operators_opt.ll -filetype=obj -mtriple=riscv64 -O2 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR} -riscv-v-vector-bits-min=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_V_LEN} -riscv-v-vector-bits-max=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_V_LEN} -target-abi=lp64f -o operators.o
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-ld -m elf64lriscv
            --defsym=operator_text_bytes=${OPERATOR_TEXT_BYTES} -T ${OPERATOR_LINKER} operators.o -o operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata operators.elf operators.image
  COMMAND ${ELF_OBJCOPY} -I binary -O elf64-littleriscv -B riscv
            --rename-section .data=.rodata,alloc,load,readonly,data,contents
            --set-section-alignment .data=16 operators.image operators.embed.o
  DEPENDS ${OUTPUT_DIR}/poly_memref.mlir
          ${OPERATOR_LINKER}
          ${DESIGN_TARGET_PLUGIN_attention}
          ${LLVM_OPT}
          ${DESIGN_COMPILER}/bin/buddy-opt
          ${DESIGN_COMPILER}/bin/buddy-translate
          ${DESIGN_COMPILER}/bin/buddy-llc
  COMMENT "Building the GP operator image"
  VERBATIM
)

# The GP operators of the reduce graphs, the image of the reduce hart's GP
# domain (_binary_reduce_operators_image_start).
math(EXPR REDUCE_OPERATOR_TEXT_BYTES "${DESIGN_PROFILES_${REDUCE_TARGET}_THREAD_GPDOMAIN_I_BUF_WORDS} * 4")
add_custom_command(
  OUTPUT reduce_operators.embed.o
  BYPRODUCTS ${OUTPUT_DIR}/reduce_operators.mlir
             ${OUTPUT_DIR}/reduce_operators_llvm.mlir
             ${OUTPUT_DIR}/reduce_operators.ll
             ${OUTPUT_DIR}/reduce_operators_opt.ll
  COMMAND ${DESIGN_TARGET_OPT_${REDUCE_TARGET}} ${OUTPUT_DIR}/reduce_acc.mlir "-symbol-privatize=exclude=operator,gp_space"
            -symbol-dce
            -verify-gp-target
            -o ${OUTPUT_DIR}/reduce_operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${OUTPUT_DIR}/reduce_operators.mlir
            -convert-vector-to-scf
            -convert-vector-to-llvm
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-math-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm=use-aligned-alloc
            -convert-func-to-llvm
            -reconcile-unrealized-casts
            -o ${OUTPUT_DIR}/reduce_operators_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/reduce_operators_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/reduce_operators.ll
  COMMAND ${LLVM_OPT} ${OUTPUT_DIR}/reduce_operators.ll -mtriple=riscv64 -mattr=${DESIGN_PROFILES_${REDUCE_TARGET}_THREAD_GPDOMAIN_LLVM_MATTR} "-passes=default<O2>" -S -o ${OUTPUT_DIR}/reduce_operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/reduce_operators_opt.ll -filetype=obj -mtriple=riscv64 -O2 -mattr=${DESIGN_PROFILES_${REDUCE_TARGET}_THREAD_GPDOMAIN_LLVM_MATTR} -riscv-v-vector-bits-min=${DESIGN_PROFILES_${REDUCE_TARGET}_THREAD_GPDOMAIN_V_LEN} -riscv-v-vector-bits-max=${DESIGN_PROFILES_${REDUCE_TARGET}_THREAD_GPDOMAIN_V_LEN} -target-abi=lp64f -o reduce_operators.o
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-ld -m elf64lriscv
            --defsym=operator_text_bytes=${REDUCE_OPERATOR_TEXT_BYTES} -T ${OPERATOR_LINKER} reduce_operators.o -o reduce_operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata reduce_operators.elf reduce_operators.image
  COMMAND ${ELF_OBJCOPY} -I binary -O elf64-littleriscv -B riscv
            --rename-section .data=.rodata,alloc,load,readonly,data,contents
            --set-section-alignment .data=16 reduce_operators.image reduce_operators.embed.o
  DEPENDS reduce.o
          ${OPERATOR_LINKER}
          ${DESIGN_TARGET_PLUGIN_${REDUCE_TARGET}}
          ${LLVM_OPT}
          ${DESIGN_COMPILER}/bin/buddy-opt
          ${DESIGN_COMPILER}/bin/buddy-translate
          ${DESIGN_COMPILER}/bin/buddy-llc
  COMMENT "Building the GP operator image"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The baremetal image of DeepSeek-R1-0528-Qwen3-8B: the runner (baremetal.cpp, every hart enters
# it), the program and the graphs, the device runtime, the bindings of the
# harts and the loaders of their threads. The parameters, the inputs and
# the PyTorch outputs of the prompt are the model image (model-pack), at the
# model region of the DDR: <elf>-ddr.bin holds it at its offset in the DDR,
# zeros before it unwritten. Both go with the workloads of the device, where
# BEMU finds them.
# ===----------------------------------------------------------------------===
math(EXPR MODEL_OFFSET "${DESIGN_DEVICE_DDR_BYTES} - ${DESIGN_DEVICE_DDR_MODEL}")
math(EXPR MODEL_IMAGE "${DESIGN_DEVICE_DDR_BASE} + ${MODEL_OFFSET}" OUTPUT_FORMAT HEXADECIMAL)
set(ELF "${DESIGN_OUTPUT}/workloads/models/poly-deepseek-r1-0528-qwen3-8b-baremetal")
add_custom_command(
  OUTPUT ${ELF}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DESIGN_OUTPUT}/workloads/models"
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++17 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -fno-exceptions -fno-rtti ${DESIGN_DEFINES} -DDEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART} -DMODEL_IMAGE=${MODEL_IMAGE}ULL -I${LIB_DIR} -I${DESIGN_DEVICE_STACK}/runtime -I${DESIGN_REPO}/stack/runtime/program -DDEVICE_CLINT_TICK_CYCLES=${DESIGN_DEVICE_CLINT_TICK_CYCLES} -c ${CMAKE_CURRENT_SOURCE_DIR}/baremetal.cpp -o baremetal.o
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany -specs=nano.specs -specs=nosys.specs -nostartfiles -Wl,-T,${BOOT_DIR}/device.ld ${DESIGN_DEFINES} -DWORKLOAD_REPORT_HART=${DESIGN_WORKLOAD_REPORT_HART} -DDEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART} -DMODEL_IMAGE=${MODEL_IMAGE}ULL -I${DESIGN_DEVICE_STACK}/runtime -I${DESIGN_REPO}/stack/runtime/program ${BOOT_DIR}/start.S ${BOOT_DIR}/syscalls.c baremetal.o program.o reduce.o poly.o operators.embed.o reduce_operators.embed.o ${DESIGN_HART_BUILD_${REDUCE_HART}}/execute.o ${DESIGN_HART_BUILD_poly}/execute.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o $<TARGET_FILE:poly-runtime> $<TARGET_FILE:runtime-program> $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:runtime> $<TARGET_FILE:kernel> -lm -lgcc -o ${ELF}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/baremetal.cpp
          ${LIB_DIR}/testutils.h
          ${BOOT_DIR}/start.S
          ${BOOT_DIR}/syscalls.c
          ${BOOT_DIR}/device.ld
          program.o
          reduce.o
          poly.o
          operators.embed.o
          reduce_operators.embed.o
          ${DESIGN_DEVICE_STACK}/runtime/device.h
          poly-runtime
          runtime-program
          runtime-baremetal
          runtime
          kernel
  COMMENT "Linking DeepSeek-R1-0528-Qwen3-8B for the poly device"
  VERBATIM
)
add_custom_command(
  OUTPUT ${ELF}-ddr.bin
  BYPRODUCTS model.img
  COMMAND ${CMAKE_BINARY_DIR}/model-pack model.img ${OUTPUT_DIR}/prompt pack-f32.data pack-i8.data inputs.data sizes.txt input-order.txt candidates.data tokens.data
  COMMAND ${CMAKE_COMMAND} -E rm -f ${ELF}-ddr.bin
  COMMAND dd if=model.img of=${ELF}-ddr.bin bs=1M oflag=seek_bytes seek=${MODEL_OFFSET} conv=sparse status=none
  DEPENDS ${OUTPUT_DIR}/prompt/pack-f32.data
          ${OUTPUT_DIR}/prompt/pack-i8.data
          ${OUTPUT_DIR}/prompt/inputs.data
          ${OUTPUT_DIR}/prompt/sizes.txt
          ${OUTPUT_DIR}/prompt/input-order.txt
          ${OUTPUT_DIR}/prompt/candidates.data
          ${OUTPUT_DIR}/prompt/tokens.data
          model-pack-tool
  COMMENT "Packing the DeepSeek-R1-0528-Qwen3-8B prompt into the model region"
  VERBATIM
)
add_library(deepseek-r1-0528-qwen3-8b-buckyball-poly-model SHARED EXCLUDE_FROM_ALL entry.cpp program.o reduce.o poly.o
  operators.embed.o reduce_operators.embed.o
)
target_link_libraries(deepseek-r1-0528-qwen3-8b-buckyball-poly-model
  PRIVATE poly-runtime-linux
)
set_target_properties(deepseek-r1-0528-qwen3-8b-buckyball-poly-model
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN;/lib"
)

get_filename_component(FLATBUFFERS_HEADER "${TOOLCHAIN}/include/flatbuffers/flatbuffers.h" REALPATH)
get_filename_component(FLATBUFFERS_INCLUDE "${FLATBUFFERS_HEADER}/../.." ABSOLUTE)
add_library(deepseek-r1-0528-qwen3-8b-buckyball-poly-runner SHARED EXCLUDE_FROM_ALL runner.cpp)
target_include_directories(deepseek-r1-0528-qwen3-8b-buckyball-poly-runner
  PRIVATE ${LIB_DIR}
          ${RUNTIME_DIR}/services
          ${FLATBUFFERS_INCLUDE}
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/runtime/include
          ${DESIGN_COMPILER}/runtime/include
)
target_link_libraries(deepseek-r1-0528-qwen3-8b-buckyball-poly-runner
  PRIVATE dl
)
set_target_properties(deepseek-r1-0528-qwen3-8b-buckyball-poly-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN;/lib"
)
configure_file(deepseek-r1-0528-qwen3-8b.rhal.mlir.in ${OUTPUT_DIR}/deepseek-r1-0528-qwen3-8b.rhal.mlir @ONLY)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/deepseek-r1-0528-qwen3-8b.rax
  COMMAND ${COMPILER_BUILD}/bin/rax-pack ${OUTPUT_DIR}/deepseek-r1-0528-qwen3-8b.rhal.mlir
          --embed-payload
          -o ${OUTPUT_DIR}/deepseek-r1-0528-qwen3-8b.rax
  DEPENDS ${OUTPUT_DIR}/deepseek-r1-0528-qwen3-8b.rhal.mlir
          ${OUTPUT_DIR}/vocab.txt
          ${OUTPUT_DIR}/input-order.txt
          ${OUTPUT_DIR}/candidates.txt
          deepseek-r1-0528-qwen3-8b-buckyball-poly-model
          deepseek-r1-0528-qwen3-8b-buckyball-poly-runner
          ${COMPILER_BUILD}/bin/rax-pack
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)
# Linux runs the RAX entry on the prompt token ids of the baremetal program
# and checks its eight steps against the same reference: the candidates of
# every step, then their tokens. Not VERBATIM: the shell redirects the
# concatenation.
file(WRITE ${OUTPUT_DIR}/run.args
  "--model\ndeepseek-r1-0528-qwen3-8b.rax\n--input\nprompt/ids.data\n--reference\nprompt/reference.data\n"
)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/prompt/reference.data
  COMMAND ${CMAKE_COMMAND} -E cat ${OUTPUT_DIR}/prompt/candidates.data ${OUTPUT_DIR}/prompt/tokens.data > ${OUTPUT_DIR}/prompt/reference.data
  DEPENDS ${OUTPUT_DIR}/prompt/candidates.data
          ${OUTPUT_DIR}/prompt/tokens.data
)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/model.img
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${OUTPUT_DIR}/prompt ${OUTPUT_DIR}/ddr
  COMMAND ${CMAKE_BINARY_DIR}/model-pack ${OUTPUT_DIR}/model.img ${OUTPUT_DIR} deepseek-r1-0528-qwen3-8b.rax prompt/ids.data prompt/reference.data
            --mapped run.args
            --mapped ddr/pack-f32.data
            --mapped ddr/pack-i8.data
  DEPENDS ${OUTPUT_DIR}/deepseek-r1-0528-qwen3-8b.rax
          ${OUTPUT_DIR}/run.args
          ${OUTPUT_DIR}/prompt/ids.data
          ${OUTPUT_DIR}/prompt/reference.data
          ${OUTPUT_DIR}/prompt/pack-f32.data
          ${OUTPUT_DIR}/prompt/pack-i8.data
          model-pack-tool
  VERBATIM
)
add_custom_target(deepseek-r1-0528-qwen3-8b-model-build DEPENDS ${ELF} ${ELF}-ddr.bin ${OUTPUT_DIR}/model.img)

# The GVN slices of this model, which `bbdev gvn-bemu --debug` builds.
add_subdirectory(gvn)
