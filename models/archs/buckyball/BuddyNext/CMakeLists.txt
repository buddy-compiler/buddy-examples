set(CMAKE_CXX_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

set(MODEL_BUDDYNEXT_DIR ${MODEL_DIR}/BuddyNext)
set(TOSA_PIPELINE "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))")

file(GLOB BUDDYNEXT_PREFILL_MLIR_FILES CONFIGURE_DEPENDS
  ${MODEL_BUDDYNEXT_DIR}/prefill/*.mlir
)
file(GLOB BUDDYNEXT_DECODE_MLIR_FILES CONFIGURE_DEPENDS
  ${MODEL_BUDDYNEXT_DIR}/decode/*.mlir
)

function(add_buddynext_buckyball_target target_name mlir_file object_name mlir_base_sanitized)
  add_custom_command(
    OUTPUT ${object_name}
    # Stage 1: linalg IR (after TOSA lowering)
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${mlir_file}
              -pass-pipeline ${TOSA_PIPELINE}
              > ${mlir_base_sanitized}_linalg.mlir
    # Stage 2: buckyball IR before lowering
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${mlir_file}
              -pass-pipeline ${TOSA_PIPELINE} |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -eliminate-empty-tensors
              -empty-tensor-to-alloc-tensor
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -one-shot-bufferize="bufferize-function-boundaries"
              -convert-linalg-to-tile
              -convert-tile-to-buckyball
              > ${mlir_base_sanitized}_buckyball.mlir
    # Stage 3: compile to object
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${mlir_file}
              -pass-pipeline ${TOSA_PIPELINE} |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -eliminate-empty-tensors
              -empty-tensor-to-alloc-tensor
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -one-shot-bufferize="bufferize-function-boundaries"
              -convert-linalg-to-tile
              -convert-tile-to-buckyball
              -lower-buckyball-to-bank-ssa
              -assign-physical-banks
              -llvm-request-c-wrappers
              -lower-bank-ssa-to-intrinsics
              -convert-trace-to-llvm="cycle-trace"
              -expand-strided-metadata
              -convert-linalg-to-loops
              -lower-buckyball
              -lower-affine
              -convert-scf-to-cf
              -convert-cf-to-llvm
              -buffer-deallocation-simplification
              -bufferization-lower-deallocations
              -convert-vector-to-llvm
              -convert-arith-to-llvm
              -convert-math-to-llvm
              -convert-math-to-libm
              -convert-func-to-llvm
              -finalize-memref-to-llvm
              -reconcile-unrealized-casts |
            ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
            ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xbbmeteor,+D -float-abi=hard -o ${object_name}
    DEPENDS ${mlir_file}
    COMMENT "Building ${target_name} object"
    VERBATIM
  )

  add_library(${target_name} STATIC ${object_name})
  set_target_properties(${target_name} PROPERTIES LINKER_LANGUAGE CXX)
  add_dependencies(${target_name} buddynext-model-build)
endfunction()

# ============================================================================
# Prefill kernels
# ============================================================================
set(BUDDYNEXT_PREFILL_TARGETS)
foreach(mlir_file ${BUDDYNEXT_PREFILL_MLIR_FILES})
  get_filename_component(mlir_base ${mlir_file} NAME_WE)
  string(REPLACE "-" "_" mlir_base_sanitized ${mlir_base})
  set(sanitized "prefill_${mlir_base_sanitized}")

  set(lib_target BuckyballBUDDYNEXT_prefill_${mlir_base_sanitized})
  set(obj_file ${sanitized}.o)
  add_buddynext_buckyball_target(${lib_target} ${mlir_file} ${obj_file} ${sanitized})

  set(run_target buddy-buckyball-buddynext-prefill-${mlir_base}-run)
  add_executable(${run_target} ${MODEL_BUDDYNEXT_DIR}/buddy-next-runtime.cpp)
  add_dependencies(${run_target} ${lib_target})
  target_link_libraries(${run_target} -static ${lib_target} ModelTestCRunnerUtils)
  list(APPEND BUDDYNEXT_PREFILL_TARGETS ${run_target})
endforeach()

# ============================================================================
# Decode kernels
# ============================================================================
set(BUDDYNEXT_DECODE_TARGETS)
foreach(mlir_file ${BUDDYNEXT_DECODE_MLIR_FILES})
  get_filename_component(mlir_base ${mlir_file} NAME_WE)
  string(REPLACE "-" "_" mlir_base_sanitized ${mlir_base})
  set(sanitized "decode_${mlir_base_sanitized}")

  set(lib_target BuckyballBUDDYNEXT_decode_${mlir_base_sanitized})
  set(obj_file ${sanitized}.o)
  add_buddynext_buckyball_target(${lib_target} ${mlir_file} ${obj_file} ${sanitized})

  set(run_target buddy-buckyball-buddynext-decode-${mlir_base}-run)
  add_executable(${run_target} ${MODEL_BUDDYNEXT_DIR}/buddy-next-runtime.cpp)
  add_dependencies(${run_target} ${lib_target})
  target_link_libraries(${run_target} -static ${lib_target} ModelTestCRunnerUtils)
  list(APPEND BUDDYNEXT_DECODE_TARGETS ${run_target})
endforeach()

add_custom_target(buddy-buckyball-buddynext-prefill-run DEPENDS ${BUDDYNEXT_PREFILL_TARGETS})
add_custom_target(buddy-buckyball-buddynext-decode-run DEPENDS ${BUDDYNEXT_DECODE_TARGETS})
add_custom_target(buddy-buckyball-buddynext-all-run
  DEPENDS buddy-buckyball-buddynext-prefill-run buddy-buckyball-buddynext-decode-run)

set(BUDDYNEXT_OUTPUT_DIR ${OUTPUT_BIN_DIR}/src/ModelTest/e2e/models/archs/buckyball/BuddyNext)

foreach(run_target ${BUDDYNEXT_PREFILL_TARGETS} ${BUDDYNEXT_DECODE_TARGETS})
  add_custom_command(TARGET ${run_target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${BUDDYNEXT_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${BUDDYNEXT_OUTPUT_DIR}/trace/cycle
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:${run_target}>
            ${BUDDYNEXT_OUTPUT_DIR}/$<TARGET_FILE_NAME:${run_target}>
    COMMENT "Copying ${run_target} executable to output directory"
    VERBATIM
  )
endforeach()
