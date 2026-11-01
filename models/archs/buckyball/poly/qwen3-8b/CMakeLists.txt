file(GLOB_RECURSE RECIPE_FRONTEND_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/frontend/*.py")

set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")
set(WEIGHTS "${MODEL_DIR}/qwen3-8b/output/qwen3-8b")

# ===----------------------------------------------------------------------===
# Import one decoder step for any number of tokens: the device program and
# the graphs of each class of core.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/program.mlir
         ${OUTPUT_DIR}/graph-reduce.mlir
         ${OUTPUT_DIR}/graph-poly.mlir
  BYPRODUCTS ${OUTPUT_DIR}/01-import.txt
             ${OUTPUT_DIR}/02-permuted.txt
             ${OUTPUT_DIR}/03-quanted.txt
             ${OUTPUT_DIR}/04-partitioned.txt
  COMMAND ${PYTHON} buddy-pytorch-importer.py ${DESIGN_PB}
  DEPENDS ${COMPILER_FRONTEND_SOURCES}
          ${RECIPE_FRONTEND_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/buddy-pytorch-importer.py
          ${DESIGN_PB}
          ${MODEL_DIR}/qwen3-8b/model.py
          ${CMAKE_CURRENT_SOURCE_DIR}/frontend/permute/graph.py
          ${WEIGHTS}/config.json
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Importing the qwen3-8b graph"
  VERBATIM
)
add_custom_target(qwen3-8b-import DEPENDS ${OUTPUT_DIR}/program.mlir)

# ===----------------------------------------------------------------------===
# The device program placed on the cores and lowered to the device runtime
# (program_placed.mlir, program_bufferized.mlir, program_lowered.mlir) by the
# device compiler of poly.
# ===----------------------------------------------------------------------===
set(DEVICE_COMPILER "${DESIGN_DEVICE_BUILD}/compiler/libdevice-compiler.so")
set(OPT ${DESIGN_COMPILER}/bin/buddy-opt --load-dialect-plugin=${DEVICE_COMPILER} --load-pass-plugin=${DEVICE_COMPILER})
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
  COMMENT "Placing and lowering the qwen3-8b program on poly"
  VERBATIM
)

# ===----------------------------------------------------------------------===
# The graphs on the poly cores through the core level of poly, by its core
# compiler, common subexpressions merged with the conversion to linalg:
# convert-linalg-to-core puts their MXINT8 quantizations and products on the
# attention or ffn threads as subgraphs launched as early as their inputs
# allow (poly_core.mlir),
# lower-frontend-fake-op computes the rest of the frontend operations
# (poly_frontend.mlir), each segment's graph inlined into it is bufferized
# (poly_bufferized.mlir) and lower-core makes the transfers DDR buffers
# and the launches batches of asynchronous engine jobs
# (poly_memref.mlir); not
# compiled further at this level. The reduce core has no hardware thread.
# ===----------------------------------------------------------------------===
set(CORE_COMPILER "${DESIGN_DEVICE_BUILD}/core/compiler/libcore-compiler.so")
set(
  CORE_OPT
  ${DESIGN_COMPILER}/bin/buddy-opt
  --load-dialect-plugin=${CORE_COMPILER}
  --load-pass-plugin=${CORE_COMPILER}
)
set(ACC_OPT ${CORE_OPT} --buckyball-design-pb=${DESIGN_PB})
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
  COMMENT "Lowering the qwen3-8b graphs to the poly cores"
  VERBATIM
)
add_custom_target(qwen3-8b-core DEPENDS ${OUTPUT_DIR}/poly_memref.mlir)
add_custom_target(qwen3-8b-model-build DEPENDS ${OUTPUT_DIR}/program_lowered.mlir)
