file(GLOB_RECURSE RECIPE_FRONTEND_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/frontend/*.py")

set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")
set(WEIGHTS "${MODEL_DIR}/llama3-2-3b-instruct/output/llama3-2-3b-instruct")
# The device compiler of poly: the device dialect, its placement and
# lowering, and lower-frontend-fake-op; built by the device stack. The program and
# the graphs of the reduce core, which has no hardware thread, are compiled
# with it.
set(DEVICE_COMPILER "${DESIGN_DEVICE_BUILD}/compiler/libdevice-compiler.so")
set(OPT ${DESIGN_COMPILER}/bin/buddy-opt --load-dialect-plugin=${DEVICE_COMPILER} --load-pass-plugin=${DEVICE_COMPILER})
set(REDUCE_TARGET ${DESIGN_DEVICE_CORES_${DESIGN_CORE_HART_reduce}_PROFILE})
set(REDUCE_OPT ${DESIGN_TARGET_OPT_${REDUCE_TARGET}})
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
# Import Llama3.2-3B-Instruct for any number of tokens: the device program and the
# graphs of each class of core.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/program.mlir
         ${OUTPUT_DIR}/vocab.txt
         ${OUTPUT_DIR}/input-order.txt
         ${OUTPUT_DIR}/graph-reduce.mlir
         ${OUTPUT_DIR}/graph-poly.mlir
         ${OUTPUT_DIR}/prompt/pack-f32.data
         ${OUTPUT_DIR}/prompt/pack-i8.data
         ${OUTPUT_DIR}/prompt/inputs.data
         ${OUTPUT_DIR}/prompt/outputs.data
         ${OUTPUT_DIR}/prompt/sizes.txt
         ${OUTPUT_DIR}/prompt/input-order.txt
  BYPRODUCTS ${OUTPUT_DIR}/01-import.txt
             ${OUTPUT_DIR}/02-permuted.txt
             ${OUTPUT_DIR}/03-quanted.txt
             ${OUTPUT_DIR}/04-partitioned.txt
  COMMAND ${PYTHON} buddy-pytorch-importer.py ${DESIGN_PB}
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          ${RECIPE_FRONTEND_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/../layout.py
          ${CMAKE_CURRENT_SOURCE_DIR}/../attention.py
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${DESIGN_PB}
          ${MODEL_DIR}/llama3-2-3b-instruct/model.py
          ${MODEL_DIR}/llama3-2-3b-instruct/input.txt
          ${WEIGHTS}/config.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing the Llama3.2-3B-Instruct graph"
  VERBATIM
)
add_custom_target(llama3-2-3b-instruct-import DEPENDS ${OUTPUT_DIR}/program.mlir)

# ===----------------------------------------------------------------------===
# The device program of Llama3.2-3B-Instruct placed on the cores and lowered to the
# device runtime (program_placed.mlir, program_bufferized.mlir,
# program_lowered.mlir).
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/program_lowered.mlir
  BYPRODUCTS ${OUTPUT_DIR}/program_placed.mlir
             ${OUTPUT_DIR}/program_bufferized.mlir
  COMMAND ${OPT} ${OUTPUT_DIR}/program.mlir -convert-linalg-to-device=design=${DESIGN_PB} -o ${OUTPUT_DIR}/program_placed.mlir
  COMMAND ${OPT} ${OUTPUT_DIR}/program_placed.mlir -convert-elementwise-to-linalg "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs hoist-dynamic-allocs" -o ${OUTPUT_DIR}/program_bufferized.mlir
  COMMAND ${OPT} ${OUTPUT_DIR}/program_bufferized.mlir -canonicalize -cse -lower-device -o ${OUTPUT_DIR}/program_lowered.mlir
  DEPENDS ${OUTPUT_DIR}/program.mlir
          ${DEVICE_COMPILER}
          ${DESIGN_PB}
  COMMENT "Placing and lowering the Llama3.2-3B-Instruct program on poly"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The graphs of Llama3.2-3B-Instruct on the poly cores through the core level, common
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
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/poly_frontend.mlir -inline -eliminate-empty-tensors -convert-elementwise-to-linalg -linalg-fuse-elementwise-ops "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -drop-equivalent-buffer-results=modify-public-functions "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs" -ownership-based-buffer-deallocation -buffer-deallocation-simplification -bufferization-lower-deallocations -canonicalize -o ${OUTPUT_DIR}/poly_bufferized.mlir
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
  COMMENT "Lowering the Llama3.2-3B-Instruct graphs to the poly cores"
  VERBATIM
)
add_custom_target(llama3-2-3b-instruct-core DEPENDS ${OUTPUT_DIR}/poly_memref.mlir)

# ===----------------------------------------------------------------------===
# The program of Llama3.2-3B-Instruct to RISC-V on the reduce core (program.ll).
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT program.o
  BYPRODUCTS ${OUTPUT_DIR}/program_llvm.mlir
             ${OUTPUT_DIR}/program.ll
             ${OUTPUT_DIR}/program_opt.ll
  COMMAND ${OPT} ${OUTPUT_DIR}/program_lowered.mlir -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts -o ${OUTPUT_DIR}/program_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/program_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/program.ll
  COMMAND ${BUDDY_SOURCE}/llvm/build/bin/opt ${OUTPUT_DIR}/program.ll -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=default<O2>" -S -o ${OUTPUT_DIR}/program_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/program_opt.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -o program.o
  DEPENDS ${OUTPUT_DIR}/program_lowered.mlir
          ${DEVICE_COMPILER}
  COMMENT "Building the Llama3.2-3B-Instruct program for poly"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The reduce core runs ordered sum chains on its accelerator;
# normalization and quantization stay on its CPU.
# lower-frontend-fake-op computes the quantization (reduce_frontend.mlir),
# each graph inlined into its function writes its outputs into the buffers
# of the program (reduce_bufferized.mlir).
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT reduce.o
  BYPRODUCTS ${OUTPUT_DIR}/reduce_linalg.mlir
             ${OUTPUT_DIR}/reduce_frontend.mlir
             ${OUTPUT_DIR}/reduce_bufferized.mlir
             ${OUTPUT_DIR}/reduce_instructions.mlir
             ${OUTPUT_DIR}/reduce_copies.mlir
             ${OUTPUT_DIR}/reduce_llvm.mlir
             ${OUTPUT_DIR}/reduce.ll
             ${OUTPUT_DIR}/reduce_opt.ll
  COMMAND ${REDUCE_OPT} ${OUTPUT_DIR}/graph-reduce.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith,canonicalize,cse))" -o ${OUTPUT_DIR}/reduce_linalg.mlir
  COMMAND ${REDUCE_OPT} ${OUTPUT_DIR}/reduce_linalg.mlir -lower-frontend-fake-op -o ${OUTPUT_DIR}/reduce_frontend.mlir
  COMMAND ${REDUCE_OPT} ${OUTPUT_DIR}/reduce_frontend.mlir -inline -eliminate-empty-tensors -forward-materialized-tensors -convert-elementwise-to-linalg -convert-linalg-to-acc -canonicalize "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -drop-equivalent-buffer-results=modify-public-functions "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs" -ownership-based-buffer-deallocation -buffer-deallocation-simplification -bufferization-lower-deallocations -canonicalize -o ${OUTPUT_DIR}/reduce_bufferized.mlir
  COMMAND ${REDUCE_OPT} ${OUTPUT_DIR}/reduce_bufferized.mlir -assign-physical-banks -lower-acc -o ${OUTPUT_DIR}/reduce_instructions.mlir
  COMMAND ${REDUCE_OPT} ${OUTPUT_DIR}/reduce_instructions.mlir -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUTPUT_DIR}/reduce_copies.mlir
  COMMAND ${REDUCE_OPT} ${OUTPUT_DIR}/reduce_copies.mlir -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm -reconcile-unrealized-casts -o ${OUTPUT_DIR}/reduce_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/reduce_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/reduce.ll
  COMMAND ${BUDDY_SOURCE}/llvm/build/bin/opt ${OUTPUT_DIR}/reduce.ll -mtriple=riscv64 -mattr=+xbuckyball,+m,+a,+f,+d,+c "-passes=default<O2>" -S -o ${OUTPUT_DIR}/reduce_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/reduce_opt.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -mattr=+xbuckyball,+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -o reduce.o
  DEPENDS ${OUTPUT_DIR}/graph-reduce.mlir
          ${DESIGN_TARGET_PLUGIN_${REDUCE_TARGET}}
  COMMENT "Building the llama3-2-3b-instruct graphs of the reduce core for poly"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The graphs of Llama3.2-3B-Instruct on the poly cores from the core level
# (poly_memref.mlir) to RISC-V: CPU control code records the Ball instruction
# streams that the core's hardware threads execute.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT poly.o
  BYPRODUCTS ${OUTPUT_DIR}/poly_copies.mlir
             ${OUTPUT_DIR}/poly_llvm.mlir
             ${OUTPUT_DIR}/poly.ll
             ${OUTPUT_DIR}/poly_opt.ll
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/poly_memref.mlir -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUTPUT_DIR}/poly_copies.mlir
  COMMAND ${CORE_OPT} ${OUTPUT_DIR}/poly_copies.mlir -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm -reconcile-unrealized-casts -o ${OUTPUT_DIR}/poly_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUTPUT_DIR}/poly_llvm.mlir
            --buddy-to-llvmir -o ${OUTPUT_DIR}/poly.ll
  COMMAND ${BUDDY_SOURCE}/llvm/build/bin/opt ${OUTPUT_DIR}/poly.ll -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=default<O2>" -S -o ${OUTPUT_DIR}/poly_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUTPUT_DIR}/poly_opt.ll -filetype=obj -relocation-model=pic -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -code-model=medium -O2 -o poly.o
  DEPENDS ${OUTPUT_DIR}/poly_memref.mlir
          ${CORE_COMPILER}
  COMMENT "Building the Llama3.2-3B-Instruct graphs of the poly cores for poly"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The baremetal image of Llama3.2-3B-Instruct: the runner (baremetal.cpp, every hart enters
# it), the program and the graphs, the device runtime, the bindings of the
# cores and the loaders of their threads. The parameters, the inputs and
# the PyTorch outputs of the prompt are the model image (model-pack), at the
# model region of the DDR: <elf>-ddr.bin holds it at its offset in the DDR,
# zeros before it unwritten. Both go with the workloads of the device, where
# BEMU finds them.
# ===----------------------------------------------------------------------===
math(EXPR MODEL_OFFSET "${DESIGN_DEVICE_DDR_BYTES} - ${DESIGN_DEVICE_DDR_MODEL}")
math(EXPR MODEL_IMAGE "${DESIGN_DEVICE_DDR_BASE} + ${MODEL_OFFSET}" OUTPUT_FORMAT HEXADECIMAL)
set(ELF "${DESIGN_OUTPUT}/workloads/models/poly-llama3-2-3b-instruct-baremetal")
add_custom_command(
  OUTPUT ${ELF}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DESIGN_OUTPUT}/workloads/models"
  COMMAND ${TOOLCHAIN}/bin/riscv64-unknown-elf-g++ -std=c++17 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -fno-exceptions -fno-rtti ${DESIGN_DEFINES} -DDEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART} -DMODEL_IMAGE=${MODEL_IMAGE}ULL -I${LIB_DIR} -I${DESIGN_DEVICE_STACK}/runtime -I${DESIGN_REPO}/stack/runtime/program -c ${CMAKE_CURRENT_SOURCE_DIR}/baremetal.cpp -o baremetal.o
  COMMAND ${ELF_CC} -g -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany -specs=nano.specs -specs=nosys.specs -nostartfiles -Wl,-T,${BOOT_DIR}/device.ld ${DESIGN_DEFINES} -DWORKLOAD_REPORT_HART=${DESIGN_WORKLOAD_REPORT_HART} -DDEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART} -DMODEL_IMAGE=${MODEL_IMAGE}ULL -I${DESIGN_DEVICE_STACK}/runtime -I${DESIGN_REPO}/stack/runtime/program ${BOOT_DIR}/start.S ${BOOT_DIR}/syscalls.c baremetal.o program.o reduce.o poly.o ${DESIGN_CORE_BUILD_reduce}/execute.o ${DESIGN_CORE_BUILD_poly}/execute.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o $<TARGET_FILE:poly-runtime> $<TARGET_FILE:runtime-program> $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:runtime> $<TARGET_FILE:engine> -lm -lgcc -o ${ELF}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/baremetal.cpp
          ${LIB_DIR}/testutils.h
          ${BOOT_DIR}/start.S
          ${BOOT_DIR}/syscalls.c
          ${BOOT_DIR}/device.ld
          program.o
          reduce.o
          poly.o
          ${DESIGN_DEVICE_STACK}/runtime/device.h
          poly-runtime
          runtime-program
          runtime-baremetal
          runtime
          engine
  COMMENT "Linking Llama3.2-3B-Instruct for the poly device"
  VERBATIM
)
add_custom_command(
  OUTPUT ${ELF}-ddr.bin
  BYPRODUCTS model.img
  COMMAND ${CMAKE_BINARY_DIR}/model-pack model.img ${OUTPUT_DIR}/prompt pack-f32.data pack-i8.data inputs.data outputs.data sizes.txt input-order.txt
  COMMAND ${CMAKE_COMMAND} -E rm -f ${ELF}-ddr.bin
  COMMAND dd if=model.img of=${ELF}-ddr.bin bs=1M oflag=seek_bytes seek=${MODEL_OFFSET} conv=sparse status=none
  DEPENDS ${OUTPUT_DIR}/prompt/pack-f32.data
          ${OUTPUT_DIR}/prompt/pack-i8.data
          ${OUTPUT_DIR}/prompt/inputs.data
          ${OUTPUT_DIR}/prompt/outputs.data
          ${OUTPUT_DIR}/prompt/sizes.txt
          ${OUTPUT_DIR}/prompt/input-order.txt
          model-pack-tool
  COMMENT "Packing the Llama3.2-3B-Instruct prompt into the model region"
  VERBATIM
)
add_library(llama3-2-3b-instruct-buckyball-poly-model SHARED EXCLUDE_FROM_ALL entry.cpp program.o reduce.o poly.o)
target_link_libraries(llama3-2-3b-instruct-buckyball-poly-model
  PRIVATE poly-runtime-linux
)
set_target_properties(llama3-2-3b-instruct-buckyball-poly-model
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN;/lib"
)

get_filename_component(FLATBUFFERS_HEADER "${TOOLCHAIN}/include/flatbuffers/flatbuffers.h" REALPATH)
get_filename_component(FLATBUFFERS_INCLUDE "${FLATBUFFERS_HEADER}/../.." ABSOLUTE)
add_library(llama3-2-3b-instruct-buckyball-poly-runner SHARED EXCLUDE_FROM_ALL runner.cpp)
target_include_directories(llama3-2-3b-instruct-buckyball-poly-runner
  PRIVATE ${LIB_DIR}
          ${RUNTIME_DIR}/services
          ${FLATBUFFERS_INCLUDE}
          ${BUDDY_SOURCE}/frontend/Interfaces
          ${BUDDY_SOURCE}/runtime/include
          ${DESIGN_COMPILER}/runtime/include
)
target_link_libraries(llama3-2-3b-instruct-buckyball-poly-runner
  PRIVATE dl
)
set_target_properties(llama3-2-3b-instruct-buckyball-poly-runner
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN;/lib"
)
configure_file(llama3-2-3b-instruct.rhal.mlir.in ${OUTPUT_DIR}/llama3-2-3b-instruct.rhal.mlir @ONLY)
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/llama3-2-3b-instruct.rax
  COMMAND ${DESIGN_COMPILER}/bin/rax-pack ${OUTPUT_DIR}/llama3-2-3b-instruct.rhal.mlir
          --embed-payload
          -o ${OUTPUT_DIR}/llama3-2-3b-instruct.rax
  DEPENDS ${OUTPUT_DIR}/llama3-2-3b-instruct.rhal.mlir
          ${OUTPUT_DIR}/vocab.txt
          ${OUTPUT_DIR}/input-order.txt
          llama3-2-3b-instruct-buckyball-poly-model
          llama3-2-3b-instruct-buckyball-poly-runner
          ${DESIGN_COMPILER}/bin/rax-pack
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)
file(READ ${MODEL_DIR}/llama3-2-3b-instruct/input.txt PROMPT)
string(STRIP "${PROMPT}" PROMPT)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${MODEL_DIR}/llama3-2-3b-instruct/input.txt)
file(WRITE ${OUTPUT_DIR}/run.args "--model\nllama3-2-3b-instruct.rax\n--prompt\n${PROMPT}\n--max-tokens\n128\n")
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/model.img
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${OUTPUT_DIR}/prompt ${OUTPUT_DIR}/ddr
  COMMAND ${CMAKE_BINARY_DIR}/model-pack ${OUTPUT_DIR}/model.img ${OUTPUT_DIR} llama3-2-3b-instruct.rax
            --mapped run.args
            --mapped ddr/pack-f32.data
            --mapped ddr/pack-i8.data
  DEPENDS ${OUTPUT_DIR}/llama3-2-3b-instruct.rax
          ${OUTPUT_DIR}/run.args
          ${OUTPUT_DIR}/prompt/pack-f32.data
          ${OUTPUT_DIR}/prompt/pack-i8.data
          model-pack-tool
  VERBATIM
)
add_custom_target(llama3-2-3b-instruct-model-build DEPENDS ${ELF} ${ELF}-ddr.bin ${OUTPUT_DIR}/model.img)
