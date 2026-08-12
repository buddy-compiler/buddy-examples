# Common host-native lowering for Buckyball models. CPU forward functions stay
# native x86; only Buckyball subgraphs are converted to the rushB C ABI.
function(add_buckyball_rushb_targets model target_prefix source_dir)
  # The Buddy external-dialect loader is process-safe only when lowering one
  # MLIR module at a time. Keep object compilation parallel, but serialize the
  # custom MLIR lowering commands even when the workload build uses ninja -j.
  set_property(GLOBAL PROPERTY JOB_POOLS buckyball_rushb_lowering=1)
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
  # Resolve a compiler subgraph to a physical Core. Chip-specific workload
  # CMake files may provide type groups parsed from their tile TOML.
  function(_buckyball_rushb_core_id source_dir source out_var)
    get_filename_component(_name ${source} NAME_WE)
    set(_id -1)
    if(_name MATCHES "^subgraph([0-9]+)$")
      set(_id ${CMAKE_MATCH_1})
    elseif(DEFINED BUCKYBALL_RUSHB_PLACEMENT_STRICT)
      file(RELATIVE_PATH _relative ${source_dir} ${source})
      get_filename_component(_relative_dir "${_relative}" DIRECTORY)
      if(NOT _relative_dir OR _relative_dir STREQUAL ".")
        string(REGEX MATCH "_([A-Za-z0-9]+)(_[A-Za-z0-9]+)*$" _suffix "${_name}")
        if(_suffix)
          string(REGEX REPLACE "^_([A-Za-z0-9]+).*$" "\\1" _type "${_suffix}")
        endif()
      else()
        string(REPLACE "/" ";" _parts "${_relative_dir}")
        list(GET _parts 0 _type)
      endif()
      set(_ids "${_POLY_CORE_IDS_${_type}}")
      if(NOT _ids)
        if(_name MATCHES "^subgraph([0-9]+)(_|$)")
          set(_id ${CMAKE_MATCH_1})
          set(_type)
          set(_ids 1)
        else()
          message(FATAL_ERROR
            "No Core type '${_type}' is declared in the Poly tile TOML for ${source}")
        endif()
      endif()
      if(NOT _type)
        set(${out_var} ${_id} PARENT_SCOPE)
        return()
      endif()
      get_property(_next GLOBAL PROPERTY BUCKYBALL_RUSHB_NEXT_${_type})
      if(NOT _next)
        set(_next 0)
      endif()
      list(LENGTH _ids _count)
      math(EXPR _slot "${_next} % ${_count}")
      list(GET _ids ${_slot} _id)
      math(EXPR _next "${_next} + 1")
      set_property(GLOBAL PROPERTY BUCKYBALL_RUSHB_NEXT_${_type} ${_next})
    endif()
    set(${out_var} ${_id} PARENT_SCOPE)
  endfunction()
  set(_rushb_host_options "-eliminate-empty-tensors;-empty-tensor-to-alloc-tensor;-convert-elementwise-to-linalg;-one-shot-bufferize='bufferize-function-boundaries';-expand-strided-metadata;-convert-linalg-to-loops;-buffer-deallocation-simplification;-bufferization-lower-deallocations;-convert-vector-to-scf;-lower-affine;-convert-scf-to-cf;-convert-cf-to-llvm;-convert-vector-to-llvm;-convert-index-to-llvm;-llvm-request-c-wrappers;-convert-arith-to-llvm;-convert-math-to-llvm;-convert-math-to-libm;-convert-func-to-llvm;-finalize-memref-to-llvm;-reconcile-unrealized-casts")
  string(REPLACE ";" " " _rushb_host_options "${_rushb_host_options}")
  foreach(mlir IN LISTS forward_mlir)
    if(IS_ABSOLUTE ${mlir})
      set(source ${mlir})
    else()
      set(source ${source_dir}/${mlir})
    endif()
    get_filename_component(name ${mlir} NAME_WE)
    # Host forward functions never execute on a Buckyball Core.
    set(rushb_core_id -1)
    set(snapshot ${mlir_snapshot_dir}/${name}.mlir)
    set(object ${CMAKE_CURRENT_BINARY_DIR}/${target_prefix}-${name}-rushb.o)
    add_custom_command(
      OUTPUT ${object}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${mlir_snapshot_dir}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${source} ${snapshot}
      COMMAND ${CMAKE_COMMAND} -E rm -f ${object} ${object}.tmp
      COMMAND bash -o pipefail -c "${BUDDY_BINARY_DIR}/buddy-opt ${snapshot} -pass-pipeline 'builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))' | ${BUDDY_BINARY_DIR}/buddy-opt ${_rushb_host_options} | ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir | ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=x86_64 -O2 -o ${object}.tmp && mv ${object}.tmp ${object}"
      DEPENDS ${BUDDY_BINARY_DIR}/buddy-opt
              ${BUDDY_BINARY_DIR}/buddy-translate
              ${BUDDY_BINARY_DIR}/buddy-llc
      JOB_POOL buckyball_rushb_lowering
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
    # Bind each compiler-generated subgraph to its tile Core. This belongs in
    # the accelerator loop: the preceding forward-function loop may be empty.
    _buckyball_rushb_core_id(${source_dir} ${source} rushb_core_id)
    set(_rushb_accel_options "-extend-trace-to-linalg;-eliminate-empty-tensors;-convert-elementwise-to-linalg;-convert-tensor-to-linalg;-one-shot-bufferize='bufferize-function-boundaries';-convert-linalg-to-tile;${BUCKYBALL_CONVERT_TILE_TO_BUCKYBALL};-extend-trace-to-buckyball;-lower-buckyball-to-bank-ssa;${BUCKYBALL_ASSIGN_PHYSICAL_BANKS};-llvm-request-c-wrappers;${BUCKYBALL_LOWER_BANK_SSA_TO_RUSHB_INTRINSICS};-lower-buckyball-intrinsics-to-rushb=core_id=${rushb_core_id};-convert-trace-to-llvm='cycle-trace';-expand-strided-metadata;-convert-linalg-to-loops;${BUCKYBALL_LOWER_BUCKYBALL_RUSHB};-lower-affine;-convert-scf-to-cf;-convert-cf-to-llvm;-convert-vector-to-scf;-convert-vector-to-llvm;-convert-index-to-llvm;-buffer-deallocation-simplification;-bufferization-lower-deallocations;-convert-math-to-llvm;-convert-math-to-libm;-convert-arith-to-llvm;-convert-func-to-llvm;-finalize-memref-to-llvm;-reconcile-unrealized-casts")
    string(REPLACE ";" " " _rushb_accel_options "${_rushb_accel_options}")
    string(REGEX REPLACE "(-convert-tile-to-buckyball=)(bank_width=[^ ]+ bank_depth=[^ ]+ bank_num=[^ ]+)" "\\1'\\2'" _rushb_accel_options "${_rushb_accel_options}")
    set(snapshot ${mlir_snapshot_dir}/${name}.mlir)
    set(object ${CMAKE_CURRENT_BINARY_DIR}/${target_prefix}-${name}-rushb.o)
    add_custom_command(
      OUTPUT ${object}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${mlir_snapshot_dir}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${source} ${snapshot}
      COMMAND ${CMAKE_COMMAND} -E rm -f ${object} ${object}.tmp
      COMMAND bash -o pipefail -c "${BUDDY_BINARY_DIR}/buddy-opt ${snapshot} -pass-pipeline 'builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))' | ${BUDDY_BINARY_DIR}/buddy-opt ${_rushb_accel_options} | ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir | ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=x86_64 -O2 -o ${object}.tmp && mv ${object}.tmp ${object}"
      DEPENDS ${BUDDY_BINARY_DIR}/buddy-opt
              ${BUDDY_BINARY_DIR}/buddy-translate
              ${BUDDY_BINARY_DIR}/buddy-llc
      JOB_POOL buckyball_rushb_lowering
      COMMENT "Lowering rushB accelerator function ${model}/${mlir}"
      VERBATIM)
    list(APPEND objects ${object})
  endforeach()

  set(codegen_target ${target_prefix}-rushB-codegen)
  add_custom_target(${codegen_target} DEPENDS ${objects})

  # Host runners follow the accelerator convention: one *-main.cpp per model
  # (BuddyNext uses buddy-next-runtime.cpp). Ignore Runner/Plugin sources used
  # by buddy-cli — those are not rushB linkable mains.
  if(model STREQUAL "BuddyNext")
    set(runner_glob "buddy-next-runtime.cpp")
  else()
    set(runner_glob "*-main.cpp")
  endif()
  file(GLOB runner_sources LIST_DIRECTORIES false
    "${MODEL_DIR}/${model}/${runner_glob}")
  list(LENGTH runner_sources runner_count)
  if(NOT runner_count EQUAL 1)
    message(FATAL_ERROR
      "rushB requires exactly one host runner matching '${runner_glob}' "
      "for ${model} (found ${runner_count}: ${runner_sources})")
  endif()
  list(GET runner_sources 0 runner_source)
  get_filename_component(runner_dir ${runner_source} DIRECTORY)
  get_filename_component(output_name ${runner_dir} NAME)
  set(output_dir ${BUCKYBALL_OUTPUT_DIR}/${output_name})
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
      set(bemu_riscv_lib_dir)
      if(backend STREQUAL "bemu")
        get_filename_component(_runtime_manifest_dir ${runtime_manifest} DIRECTORY)
        file(GLOB _bemu_riscv_lib_dirs LIST_DIRECTORIES true
          "${_runtime_manifest_dir}/target/release/build/bemu-goban-*/out/spike_install/lib")
        list(LENGTH _bemu_riscv_lib_dirs _bemu_riscv_lib_dir_count)
        if(_bemu_riscv_lib_dir_count GREATER 0)
          list(GET _bemu_riscv_lib_dirs 0 bemu_riscv_lib_dir)
        endif()
      endif()
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
                ${bemu_riscv_lib_dir}/libriscv.so
                -Wl,-rpath,${output_dir}
                -Wl,-rpath,${bemu_riscv_lib_dir}
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
