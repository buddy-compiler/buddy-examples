set(CMAKE_CXX_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

set(MODEL_BUDDYNEXT_DIR ${MODEL_DIR}/BuddyNext)
file(GLOB BUDDYNEXT_MLIR_FILES CONFIGURE_DEPENDS
  ${MODEL_BUDDYNEXT_DIR}/*.mlir
)

function(add_buddynext_gemmini_target target_name mlir_file object_name)
  add_custom_command(
    OUTPUT ${object_name}
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${mlir_file}
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -eliminate-empty-tensors
              -empty-tensor-to-alloc-tensor
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -one-shot-bufferize="bufferize-function-boundaries"
              -convert-linalg-to-gemmini
              -expand-strided-metadata
              -convert-linalg-to-loops
              -convert-scf-to-cf
              -llvm-request-c-wrappers
              -lower-gemmini
              -ownership-based-buffer-deallocation
              -buffer-deallocation-simplification
              -bufferization-lower-deallocations
              -convert-arith-to-llvm
              -convert-math-to-llvm
              -convert-math-to-libm
              -convert-func-to-llvm
              -finalize-memref-to-llvm
              -reconcile-unrealized-casts |
            ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
            ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard -o ${object_name}
    DEPENDS ${mlir_file}
    COMMENT "Building ${target_name} object"
    VERBATIM
  )

  add_library(${target_name} STATIC ${object_name})
  set_target_properties(${target_name} PROPERTIES LINKER_LANGUAGE CXX)
  add_dependencies(${target_name} buddynext-model-build)
endfunction()

set(BUDDYNEXT_RUN_TARGETS)
foreach(mlir_file ${BUDDYNEXT_MLIR_FILES})
  get_filename_component(mlir_base ${mlir_file} NAME_WE)
  string(REPLACE "-" "_" mlir_base_sanitized ${mlir_base})

  set(lib_target GemminiBUDDYNEXT_${mlir_base_sanitized})
  set(obj_file ${mlir_base_sanitized}.o)
  add_buddynext_gemmini_target(
    ${lib_target}
    ${mlir_file}
    ${obj_file}
  )

  set(run_target buddy-gemmini-buddynext-${mlir_base}-run)
  add_executable(${run_target} ${MODEL_BUDDYNEXT_DIR}/buddy-next-runtime.cpp)
  add_dependencies(${run_target} ${lib_target})
  target_link_libraries(
    ${run_target}
    ${lib_target}
    ModelTestCRunnerUtils
  )
  list(APPEND BUDDYNEXT_RUN_TARGETS ${run_target})
endforeach()

add_custom_target(buddy-gemmini-buddynext-all-run DEPENDS ${BUDDYNEXT_RUN_TARGETS})
