# Select exactly one Buckyball chip layout for this workload build.
# Model↔chip is many-to-many: a chip may host many models, and the same
# model name may appear under multiple chips. Layout dirs under this folder
# are the source of truth for which pairs exist today.
if(DEFINED BUCKYBALL_WORKLOAD_CHIP)
  set(BUCKYBALL_ARCH "${BUCKYBALL_WORKLOAD_CHIP}")
else()
  set(BUCKYBALL_ARCH "" CACHE STRING
      "Buckyball chip layout directory under archs/buckyball/")
endif()

set(BUCKYBALL_DIR ${CMAKE_CURRENT_LIST_DIR}/${BUCKYBALL_ARCH})
if(BUCKYBALL_ARCH STREQUAL "" OR NOT EXISTS ${BUCKYBALL_DIR})
  message(FATAL_ERROR
    "No Buckyball ModelTest layout for chip '${BUCKYBALL_ARCH}' "
    "(expected directory ${CMAKE_CURRENT_LIST_DIR}/<chip>)")
endif()

if(BUCKYBALL_ARCH STREQUAL "goban" OR BUCKYBALL_ARCH STREQUAL "pebble")
  list(FIND BUCKYBALL_TARGETS "${BUCKYBALL_ARCH}" BUCKYBALL_MODEL_TARGET_INDEX)
  if(BUCKYBALL_MODEL_TARGET_INDEX EQUAL -1)
    message(FATAL_ERROR
      "Model layout ${BUCKYBALL_ARCH} requires compiler target ${BUCKYBALL_ARCH}")
  endif()
  set(BUCKYBALL_MODEL_TARGET ${BUCKYBALL_ARCH})
  set(BUCKYBALL_CONVERT_TILE_TO_BUCKYBALL
    --target=${BUCKYBALL_MODEL_TARGET}
    -convert-tile-to-buckyball)
  set(BUCKYBALL_ASSIGN_PHYSICAL_BANKS -assign-physical-banks)
endif()

set(BUCKYBALL_OUTPUT_DIR
    ${OUTPUT_BIN_DIR}/src/ModelTest/e2e/models/archs/buckyball/${BUCKYBALL_ARCH})
set(BUCKYBALL_LENET_DIR ${BUCKYBALL_DIR}/LeNet)
set(BUCKYBALL_RESNET18_DIR ${BUCKYBALL_DIR}/ResNet18)
set(BUCKYBALL_MOBILENETV3_DIR ${BUCKYBALL_DIR}/MobileNetV3)
set(BUCKYBALL_BERT_DIR ${BUCKYBALL_DIR}/Bert)
set(BUCKYBALL_STABLEDIFFUSION_DIR ${BUCKYBALL_DIR}/StableDiffusion)
set(BUCKYBALL_LLAMA2_DIR ${BUCKYBALL_DIR}/llama2)
set(BUCKYBALL_DEEPSEEKR1_DIR ${BUCKYBALL_DIR}/DeepSeekR1)
set(BUCKYBALL_QWEN3_DIR ${BUCKYBALL_DIR}/Qwen3)
set(BUCKYBALL_YOLO26_DIR ${BUCKYBALL_DIR}/YOLO26)
set(BUCKYBALL_BUDDYNEXT_DIR ${BUCKYBALL_DIR}/BuddyNext)
set(BUCKYBALL_GEMMA4_DIR ${BUCKYBALL_DIR}/Gemma4)
set(BUCKYBALL_WHISPER_DIR ${BUCKYBALL_DIR}/Whisper)

file(GLOB BUCKYBALL_LAYOUT_ENTRIES LIST_DIRECTORIES true
  RELATIVE ${BUCKYBALL_DIR} ${BUCKYBALL_DIR}/*)
set(BUCKYBALL_ALLOWED_MODELS)
foreach(entry IN LISTS BUCKYBALL_LAYOUT_ENTRIES)
  if(IS_DIRECTORY ${BUCKYBALL_DIR}/${entry})
    list(APPEND BUCKYBALL_ALLOWED_MODELS ${entry})
  endif()
endforeach()

# Explicit -DMODEL= must match a layout on this chip. Default-all MODEL_*
# without -DMODEL= only builds layouts that exist (many-to-many; no 1:1 gate).
if(DEFINED MODEL)
  set(BUCKYBALL_ALL_MODELS
      LENET RESNET18 MOBILENETV3 YOLO26 BUDDYNEXT DEEPSEEKR1 STABLEDIFFUSION
      WHISPER LLAMA2 BERT GEMMA4 QWEN3 MINIMAX_H3_FL2VA MINIMAX_H3_REF2VA)
  foreach(model_upper IN LISTS BUCKYBALL_ALL_MODELS)
    if(MODEL_${model_upper})
      set(model_allowed FALSE)
      foreach(allowed_model IN LISTS BUCKYBALL_ALLOWED_MODELS)
        string(TOUPPER "${allowed_model}" allowed_model_upper)
        if(allowed_model_upper STREQUAL model_upper)
          set(model_allowed TRUE)
        endif()
      endforeach()
      if(NOT model_allowed)
        message(FATAL_ERROR
          "Model '${model_upper}' is not supported by Buckyball chip '${BUCKYBALL_ARCH}' "
          "(layout dirs: ${BUCKYBALL_ALLOWED_MODELS})")
      endif()
    endif()
  endforeach()
endif()

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
  # A subgraph names its compiler profile (`subgraph0_prefill.mlir`).  Resolve
  # that profile to one physical Core from the Chip.pb-derived mapping.
  function(_buckyball_rushb_target source out_core_var out_target_var)
    get_filename_component(_name ${source} NAME_WE)
    list(LENGTH BUCKYBALL_TARGETS _target_count)
    if(_target_count EQUAL 1)
      list(GET BUCKYBALL_TARGETS 0 _target)
    elseif(_name MATCHES "^subgraph[0-9]+_([A-Za-z][A-Za-z0-9]*)(_|$)")
      set(_target "${CMAKE_MATCH_1}")
    else()
      message(FATAL_ERROR
        "${source}: heterogeneous accelerator subgraphs must be named subgraph<N>_<target>.mlir")
    endif()
    list(FIND BUCKYBALL_TARGETS "${_target}" _target_index)
    if(_target_index EQUAL -1)
      message(FATAL_ERROR
        "${source}: subgraph target '${_target}' is not a Chip.pb compiler profile")
    endif()

    set(_ids)
    foreach(_core_target IN LISTS BUCKYBALL_CORE_TARGET_LINES)
      string(REPLACE ":" ";" _core_target_fields "${_core_target}")
      list(LENGTH _core_target_fields _core_target_field_count)
      if(NOT _core_target_field_count EQUAL 2)
        message(FATAL_ERROR "invalid Chip.pb core target entry: ${_core_target}")
      endif()
      list(GET _core_target_fields 0 _core_id)
      list(GET _core_target_fields 1 _core_target_name)
      if(NOT _core_id MATCHES "^[0-9]+$" OR _core_target_name STREQUAL "")
        message(FATAL_ERROR "invalid Chip.pb core target entry: ${_core_target}")
      endif()
      if("${_core_target_name}" STREQUAL "${_target}")
        list(APPEND _ids "${_core_id}")
      endif()
    endforeach()
    list(LENGTH _ids _count)
    if(_count EQUAL 0)
      message(FATAL_ERROR
        "${source}: Chip.pb has no physical Core for target '${_target}'")
    endif()
    get_property(_next GLOBAL PROPERTY BUCKYBALL_RUSHB_NEXT_${_target})
    if(NOT _next)
      set(_next 0)
    endif()
    math(EXPR _slot "${_next} % ${_count}")
    list(GET _ids ${_slot} _core_id)
    math(EXPR _next "${_next} + 1")
    set_property(GLOBAL PROPERTY BUCKYBALL_RUSHB_NEXT_${_target} ${_next})
    set(${out_core_var} ${_core_id} PARENT_SCOPE)
    set(${out_target_var} ${_target} PARENT_SCOPE)
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
    _buckyball_rushb_target(${source} rushb_core_id rushb_target)
    set(_rushb_accel_options "--target=${rushb_target};-extend-trace-to-linalg;-eliminate-empty-tensors;-convert-elementwise-to-linalg;-convert-tensor-to-linalg;-one-shot-bufferize='bufferize-function-boundaries';-convert-linalg-to-tile;-convert-tile-to-buckyball;-extend-trace-to-buckyball;-lower-buckyball-to-bank-ssa;-assign-physical-banks;-llvm-request-c-wrappers;${BUCKYBALL_LOWER_BANK_SSA_TO_RUSHB_INTRINSICS};-lower-buckyball-intrinsics-to-rushb=core_id=${rushb_core_id};-convert-trace-to-llvm='cycle-trace';-expand-strided-metadata;-convert-linalg-to-loops;${BUCKYBALL_LOWER_BUCKYBALL_RUSHB};-lower-affine;-convert-scf-to-cf;-convert-cf-to-llvm;-convert-vector-to-scf;-convert-vector-to-llvm;-convert-index-to-llvm;-buffer-deallocation-simplification;-bufferization-lower-deallocations;-convert-math-to-llvm;-convert-math-to-libm;-convert-arith-to-llvm;-convert-func-to-llvm;-finalize-memref-to-llvm;-reconcile-unrealized-casts")
    string(REPLACE ";" " " _rushb_accel_options "${_rushb_accel_options}")
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

function(add_buckyball_perfetto_target target)
  set(one_value_args TRACE_DIR TRACE_TOML OUTPUT)
  set(multi_value_args MLIR)
  cmake_parse_arguments(PFT "" "${one_value_args}" "${multi_value_args}" ${ARGN})
  if(NOT PFT_TRACE_DIR OR NOT PFT_TRACE_TOML OR NOT PFT_MLIR)
    message(FATAL_ERROR "${target} requires TRACE_DIR, TRACE_TOML, and MLIR")
  endif()
  if(PFT_OUTPUT)
    set(output ${PFT_OUTPUT})
  else()
    set(output ${PFT_TRACE_DIR}/perfetto.json)
  endif()
  set(mlir_args)
  foreach(mlir IN LISTS PFT_MLIR)
    list(APPEND mlir_args --mlir ${mlir})
  endforeach()
  add_custom_target(${target}
    COMMAND ${Python3_EXECUTABLE}
            ${WORKLOAD_DIR}/../framework/trace/perfetto.py
            ${PFT_TRACE_DIR} ${PFT_TRACE_TOML} -o ${output} ${mlir_args}
    COMMENT "Generating Perfetto trace ${output}"
    VERBATIM USES_TERMINAL)
endfunction()

if(MODEL_LENET AND EXISTS ${BUCKYBALL_DIR}/LeNet)
  add_subdirectory(${BUCKYBALL_ARCH}/LeNet)
  add_buckyball_rushb_targets(LeNet buddy-buckyball-lenet ${MODEL_DIR}/LeNet
    forward.mlir subgraph0.mlir)
endif()
if(MODEL_RESNET18 AND EXISTS ${BUCKYBALL_DIR}/ResNet18)
  add_subdirectory(${BUCKYBALL_ARCH}/ResNet18)
  add_buckyball_rushb_targets(ResNet18 buddy-buckyball-resnet ${MODEL_DIR}/ResNet18
    forward.mlir subgraph0.mlir)
endif()
if(MODEL_MOBILENETV3 AND EXISTS ${BUCKYBALL_DIR}/MobileNetV3)
  add_subdirectory(${BUCKYBALL_ARCH}/MobileNetV3)
  add_buckyball_rushb_targets(MobileNetV3 buddy-buckyball-mobilenetv3
    ${MODEL_DIR}/MobileNetV3 forward.mlir subgraph0.mlir)
endif()
if(MODEL_YOLO26 AND EXISTS ${BUCKYBALL_DIR}/YOLO26)
  add_subdirectory(${BUCKYBALL_ARCH}/YOLO26)
  add_buckyball_rushb_targets(YOLO26 buddy-buckyball-yolo26 ${MODEL_DIR}/YOLO26
    forward.mlir subgraph0.mlir)
endif()
if(MODEL_BUDDYNEXT AND EXISTS ${BUCKYBALL_DIR}/BuddyNext)
  add_subdirectory(${BUCKYBALL_ARCH}/BuddyNext)
  file(GLOB BUDDYNEXT_RUSHB_MLIR CONFIGURE_DEPENDS
    ${MODEL_DIR}/BuddyNext/prefill/*.mlir ${MODEL_DIR}/BuddyNext/decode/*.mlir)
  add_buckyball_rushb_targets(BuddyNext buddy-buckyball-buddynext
    ${MODEL_DIR}/BuddyNext ${BUDDYNEXT_RUSHB_MLIR})
endif()
if(MODEL_DEEPSEEKR1 AND EXISTS ${BUCKYBALL_DIR}/DeepSeekR1)
  add_subdirectory(${BUCKYBALL_ARCH}/DeepSeekR1)
  add_buckyball_rushb_targets(DeepSeekR1 buddy-buckyball-deepseekr1
    ${MODEL_DIR}/DeepSeekR1 forward.mlir subgraph0.mlir)
endif()
if(MODEL_STABLEDIFFUSION AND EXISTS ${BUCKYBALL_DIR}/StableDiffusion)
  add_subdirectory(${BUCKYBALL_ARCH}/StableDiffusion)
  add_buckyball_rushb_targets(StableDiffusion buddy-buckyball-stable-diffusion
    ${MODEL_DIR}/StableDiffusion forward_text_encoder.mlir
    subgraph0_text_encoder.mlir forward_unet.mlir subgraph0_unet.mlir
    forward_vae.mlir subgraph0_vae.mlir)
endif()
if(MODEL_WHISPER AND EXISTS ${BUCKYBALL_DIR}/Whisper)
  add_subdirectory(${BUCKYBALL_ARCH}/Whisper)
  add_buckyball_rushb_targets(Whisper buddy-buckyball-whisper
    ${CMAKE_BINARY_DIR}/workloads/src/ModelTest/e2e/models/models/Whisper
    forward.mlir subgraph0.mlir)
endif()
if(MODEL_LLAMA2 AND EXISTS ${BUCKYBALL_DIR}/llama2)
  add_subdirectory(${BUCKYBALL_ARCH}/llama2)
  add_buckyball_rushb_targets(llama2 buddy-buckyball-llama2 ${MODEL_DIR}/llama2
    forward.mlir subgraph0.mlir)
endif()
if(MODEL_BERT AND EXISTS ${BUCKYBALL_DIR}/Bert)
  add_subdirectory(${BUCKYBALL_ARCH}/Bert)
  add_buckyball_rushb_targets(Bert buddy-buckyball-bert ${MODEL_DIR}/Bert
    forward.mlir subgraph0.mlir subgraph1.mlir subgraph2.mlir subgraph3.mlir)
endif()
if(MODEL_GEMMA4 AND EXISTS ${BUCKYBALL_DIR}/Gemma4)
  add_subdirectory(${BUCKYBALL_ARCH}/Gemma4)
  add_buckyball_rushb_targets(Gemma4 buddy-buckyball-gemma4 ${MODEL_DIR}/Gemma4
    forward_prefill.mlir subgraph0_prefill.mlir
    forward_decode.mlir subgraph0_decode.mlir)
endif()
if(MODEL_QWEN3 AND EXISTS ${BUCKYBALL_DIR}/Qwen3)
  add_subdirectory(${BUCKYBALL_ARCH}/Qwen3)
  add_buckyball_rushb_targets(Qwen3 buddy-buckyball-qwen3 ${MODEL_DIR}/Qwen3
    forward_prefill_0_6b.mlir subgraph0_prefill_0_6b.mlir
    forward_decode_0_6b.mlir subgraph0_decode_0_6b.mlir)
  add_dependencies(buddy-buckyball-qwen3-rushB-codegen qwen3-model-build)
endif()
