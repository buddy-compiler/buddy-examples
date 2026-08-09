# Common host-native lowering for Buckyball models. CPU forward functions stay
# native x86; only Buckyball subgraphs are converted to the rushB C ABI.
function(add_buckyball_rushb_targets model target_prefix source_dir)
  set(interface_dir ${BUDDY_MLIR_DIR}/frontend/Interfaces)
  set(dip_mlir ${BUDDY_MLIR_DIR}/frontend/Interfaces/lib/DIP.mlir)
  set(dap_extend_mlir ${BUDDY_MLIR_DIR}/frontend/Interfaces/lib/DAP-extend.mlir)
  set(forward_mlir ${ARGN})
  set(subgraph_mlir)
  foreach(mlir IN LISTS forward_mlir)
    if(model STREQUAL "BuddyNext" OR mlir MATCHES "(^|/)subgraph")
      list(APPEND subgraph_mlir ${mlir})
    endif()
  endforeach()
  if(model STREQUAL "BuddyNext")
    set(forward_mlir)
  else()
    list(FILTER forward_mlir EXCLUDE REGEX "(^|/)subgraph")
  endif()

  set(objects)
  set(mlir_snapshot_dir ${CMAKE_CURRENT_BINARY_DIR}/rushB/mlir/${target_prefix})
  foreach(mlir IN LISTS forward_mlir)
    if(IS_ABSOLUTE ${mlir})
      set(source ${mlir})
    else()
      set(source ${source_dir}/${mlir})
    endif()
    get_filename_component(name ${mlir} NAME_WE)
    set(snapshot ${mlir_snapshot_dir}/${name}.mlir)
    set(object ${CMAKE_CURRENT_BINARY_DIR}/${target_prefix}-${name}-rushb.o)
    add_custom_command(
      OUTPUT ${object}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${mlir_snapshot_dir}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${source} ${snapshot}
      COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${snapshot}
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
              ${BUDDY_BINARY_DIR}/buddy-opt
              -eliminate-empty-tensors
              -empty-tensor-to-alloc-tensor
              -convert-elementwise-to-linalg
              -one-shot-bufferize="bufferize-function-boundaries"
              -expand-strided-metadata
              -convert-linalg-to-loops
              -buffer-deallocation-simplification
              -bufferization-lower-deallocations
              -convert-vector-to-scf
              -lower-affine
              -convert-scf-to-cf
              -convert-cf-to-llvm
              -convert-vector-to-llvm
              -llvm-request-c-wrappers
              -convert-arith-to-llvm
              -convert-math-to-llvm
              -convert-math-to-libm
              -convert-func-to-llvm
              -finalize-memref-to-llvm
              -reconcile-unrealized-casts |
              ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
              ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=x86_64 -O2 -o ${object}
      DEPENDS ${BUDDY_BINARY_DIR}/buddy-opt
              ${BUDDY_BINARY_DIR}/buddy-translate
              ${BUDDY_BINARY_DIR}/buddy-llc
      COMMENT "Lowering rushB host CPU function ${model}/${mlir}"
      VERBATIM)
    list(APPEND objects ${object})
  endforeach()

  foreach(mlir IN LISTS subgraph_mlir)
    if(IS_ABSOLUTE ${mlir})
      set(source ${mlir})
    else()
      set(source ${source_dir}/${mlir})
    endif()
    get_filename_component(name ${mlir} NAME_WE)
    set(snapshot ${mlir_snapshot_dir}/${name}.mlir)
    set(object ${CMAKE_CURRENT_BINARY_DIR}/${target_prefix}-${name}-rushb.o)
    add_custom_command(
      OUTPUT ${object}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${mlir_snapshot_dir}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${source} ${snapshot}
      COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${snapshot}
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
              ${BUDDY_BINARY_DIR}/buddy-opt
              -eliminate-empty-tensors
              -empty-tensor-to-alloc-tensor
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -one-shot-bufferize="bufferize-function-boundaries"
              -convert-linalg-to-tile
              -convert-tile-to-buckyball
              -batchmatmul-optimize
              -lower-buckyball-to-bank-ssa
              -assign-physical-banks
              -llvm-request-c-wrappers
              ${BUCKYBALL_LOWER_BANK_SSA_TO_RUSHB_INTRINSICS}
              -convert-trace-to-llvm="cycle-trace"
              -expand-strided-metadata
              -convert-linalg-to-loops
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
              ${BUCKYBALL_LOWER_BUCKYBALL_RUSHB}
              -lower-buckyball-intrinsics-to-rushb
              -reconcile-unrealized-casts |
              ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
              ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=x86_64 -O2 -o ${object}
      DEPENDS ${BUDDY_BINARY_DIR}/buddy-opt
              ${BUDDY_BINARY_DIR}/buddy-translate
              ${BUDDY_BINARY_DIR}/buddy-llc
      COMMENT "Lowering rushB accelerator function ${model}/${mlir}"
      VERBATIM)
    list(APPEND objects ${object})
  endforeach()

  set(codegen_target ${target_prefix}-rushB-codegen)
  add_custom_target(${codegen_target} DEPENDS ${objects})

  file(GLOB runner_sources LIST_DIRECTORIES false "${MODEL_DIR}/${model}/*.cpp")
  list(LENGTH runner_sources runner_count)
  if(NOT runner_count EQUAL 1)
    message(FATAL_ERROR "rushB requires exactly one host runner for ${model}")
  endif()
  list(GET runner_sources 0 runner_source)
  get_filename_component(runner_dir ${runner_source} DIRECTORY)
  get_filename_component(output_name ${runner_dir} NAME)
  set(output_dir ${OUTPUT_BIN_DIR}/src/ModelTest/e2e/models/archs/buckyball/${output_name})
  set(runtime_source ${BUCKYBALL_REPO_ROOT}/compiler/lib/RushBRuntime.c)
  set(dma_runtime_source ${WORKLOAD_LIB_DIR}/bbhw/mem/mem.c)
  set(runtime_objects)

  set(crunner_object ${CMAKE_CURRENT_BINARY_DIR}/${target_prefix}-rushb-crunner-utils.o)
  add_custom_command(
    OUTPUT ${crunner_object}
    COMMAND c++ -std=c++17 -O2
            -I${BUDDY_MLIR_DIR}/llvm/mlir/include/mlir/ExecutionEngine
            -c ${MODELTEST_LIB_DIR}/CRunnerUtils.cpp -o ${crunner_object}
    DEPENDS ${MODELTEST_LIB_DIR}/CRunnerUtils.cpp
    COMMENT "Building rushB host runtime support for ${model}"
    VERBATIM)
  list(APPEND runtime_objects ${crunner_object})
  list(APPEND runtime_objects ${dma_runtime_source})

  if(model STREQUAL "MobileNetV3" OR model STREQUAL "ResNet18" OR
     model STREQUAL "YOLO26")
    set(dip_object ${CMAKE_CURRENT_BINARY_DIR}/${target_prefix}-rushb-dip.o)
    add_custom_command(
      OUTPUT ${dip_object}
      COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${dip_mlir}
              -lower-dip
              -arith-expand
              -lower-affine
              -convert-scf-to-cf
              -convert-math-to-llvm
              -convert-math-to-libm
              -convert-vector-to-llvm
              -finalize-memref-to-llvm
              -convert-func-to-llvm
              -convert-cf-to-llvm
              -convert-arith-to-llvm
              -reconcile-unrealized-casts |
              ${BUDDY_BINARY_DIR}/buddy-translate --mlir-to-llvmir |
              ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=x86_64 -O2
              -o ${dip_object}
      DEPENDS ${dip_mlir}
              ${BUDDY_BINARY_DIR}/buddy-opt
              ${BUDDY_BINARY_DIR}/buddy-translate
              ${BUDDY_BINARY_DIR}/buddy-llc
      COMMENT "Building rushB host DIP support for ${model}"
      VERBATIM)
    list(APPEND runtime_objects ${dip_object})
  endif()

  if(model STREQUAL "Whisper")
    set(dap_object ${CMAKE_CURRENT_BINARY_DIR}/${target_prefix}-rushb-dap-extend.o)
    add_custom_command(
      OUTPUT ${dap_object}
      COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${dap_extend_mlir}
              -extend-dap
              -one-shot-bufferize
              -convert-linalg-to-loops
              -convert-scf-to-cf
              -expand-strided-metadata
              -lower-affine
              -convert-vector-to-llvm
              -memref-expand
              -arith-expand
              -convert-arith-to-llvm
              -finalize-memref-to-llvm
              -convert-math-to-llvm
              -llvm-request-c-wrappers
              -convert-func-to-llvm
              -convert-cf-to-llvm
              -reconcile-unrealized-casts |
              ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
              ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=x86_64 -O2
              -o ${dap_object}
      DEPENDS ${dap_extend_mlir}
              ${BUDDY_BINARY_DIR}/buddy-opt
              ${BUDDY_BINARY_DIR}/buddy-translate
              ${BUDDY_BINARY_DIR}/buddy-llc
      COMMENT "Building rushB host DAP support for ${model}"
      VERBATIM)
    list(APPEND runtime_objects ${dap_object})
  endif()

  set(runner_definitions)
  if(model STREQUAL "Gemma4")
    list(APPEND runner_definitions
      -DGEMMA4_EXAMPLE_PATH=\"${MODEL_DIR}/Gemma4/\"
      -DGEMMA4_EXAMPLE_BUILD_PATH=\"${output_dir}/\")
  elseif(model STREQUAL "Qwen3")
    list(APPEND runner_definitions
      -DQWEN3_0_6B_EXAMPLE_PATH=\"${MODEL_DIR}/Qwen3/\"
      -DQWEN3_0_6B_EXAMPLE_BUILD_PATH=\"${output_dir}/\")
  endif()

  foreach(backend bemu verilator)
    set(target ${target_prefix}-rushB-${backend}-run)
    if(NOT TARGET ${target})
      set(build_dir ${CMAKE_CURRENT_BINARY_DIR}/rushB/${target})
      set(binary ${build_dir}/${target}.bin)
      if(backend STREQUAL "bemu")
        set(runtime_manifest ${BUCKYBALL_RUSHB_BEMU_MANIFEST})
        set(runtime_library ${BUCKYBALL_RUSHB_BEMU_LIBRARY})
        set(runtime_name bebop_bemu)
        set(runtime_build COMMAND cargo build --release --manifest-path ${runtime_manifest} --lib)
        set(runtime_dependency ${runtime_manifest})
      else()
        set(runtime_library ${BUCKYBALL_RUSHB_VERILATOR_LIBRARY})
        set(runtime_name bebop_verilator)
        set(runtime_build)
        set(runtime_dependency ${runtime_library})
      endif()
      set(local_library ${build_dir}/lib${runtime_name}.so)
      set(output_binary ${output_dir}/${target_prefix}-rushB-${backend}-run)
      set(output_library ${output_dir}/lib${runtime_name}.so)
      add_custom_command(
        OUTPUT ${binary} ${output_binary} ${output_library}
        BYPRODUCTS ${local_library}
        ${runtime_build}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${build_dir}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${runtime_library} ${local_library}
        COMMAND c++ -no-pie -std=c++17 -O2
                -I${interface_dir}
                -I${MODELTEST_LIB_DIR}
                -I${WORKLOAD_LIB_DIR}
                -I${BUCKYBALL_REPO_ROOT}/compiler/include
                ${runner_definitions}
                ${runner_source} ${runtime_source} ${objects} ${runtime_objects}
                -L${build_dir} -l${runtime_name}
                -Wl,-rpath,${output_dir}
                -o ${binary}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${output_dir}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${output_dir}/trace/cycle
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${source_dir} ${output_dir}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${binary}
                ${output_binary}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${local_library}
                ${output_library}
        DEPENDS ${objects} ${runtime_objects}
                ${runner_source} ${runtime_source}
                ${BUCKYBALL_REPO_ROOT}/compiler/include/buckyball/rushb.h
                ${runtime_dependency}
        COMMENT "Building rushB ${backend} ${model}"
        VERBATIM)
      add_custom_target(${target}
        DEPENDS ${binary} ${output_binary} ${output_library})
      add_dependencies(${target} ${codegen_target})
    endif()
  endforeach()
endfunction()
