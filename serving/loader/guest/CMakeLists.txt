cmake_minimum_required(VERSION 3.22)

# Build guest executables independently of the image builder's toolchain.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(BuckyballWorkerMux LANGUAGES CXX)
  find_package(Threads REQUIRED)
  add_executable(worker-mux mux.cpp)
  target_compile_features(worker-mux PRIVATE cxx_std_17)
  target_compile_definitions(worker-mux PRIVATE _GNU_SOURCE)
  target_compile_options(worker-mux PRIVATE -Wall -Wextra -Werror)
  target_link_libraries(worker-mux PRIVATE Threads::Threads)
  target_link_options(worker-mux PRIVATE -static)
  install(TARGETS worker-mux RUNTIME DESTINATION bin)
  add_executable(resource-check resource-check.cpp)
  target_compile_features(resource-check PRIVATE cxx_std_17)
  target_include_directories(resource-check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime/include)
  target_compile_definitions(resource-check PRIVATE _GNU_SOURCE)
  target_compile_options(resource-check PRIVATE -Wall -Wextra -Werror)
  target_link_options(resource-check PRIVATE -static)
  install(TARGETS resource-check RUNTIME DESTINATION bin)
  return()
endif()

# Configure the serving payload installed by the kernel image builder.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${MODEL_OUTPUT_DIR}/layout.json)
file(READ ${MODEL_OUTPUT_DIR}/layout.json MODEL_LAYOUT)
string(JSON MODEL_GUEST ERROR_VARIABLE MODEL_GUEST_ERROR GET "${MODEL_LAYOUT}" execution p2e)
if(NOT MODEL_GUEST_ERROR)
  string(JSON MODEL_EXECUTION GET "${MODEL_GUEST}" kind)
else()
  string(JSON MODEL_EXECUTION GET "${MODEL_LAYOUT}" execution kind)
endif()
if(MODEL_EXECUTION STREQUAL "host-worker")
  string(JSON WORKER_COUNT GET "${MODEL_GUEST}" workers)
  string(JSON WORKER_PROTOCOL GET "${MODEL_GUEST}" protocol)
  if(WORKER_COUNT LESS 1 OR NOT WORKER_PROTOCOL STREQUAL "bbmux1")
    message(FATAL_ERROR "host-worker requires positive workers and protocol=bbmux1")
  endif()
  if(BUCKYBALL_HART_COUNT LESS WORKER_COUNT)
    message(FATAL_ERROR "host-worker requires at least ${WORKER_COUNT} Linux-visible CPUs")
  endif()
  set(WORKER_CPU_SCRIPT ${REPO_ROOT}/stack/serving/loader/guest/cpus.py)
  set(WORKER_CHIP_PB ${REPO_ROOT}/examples/chips/${BUCKYBALL_KERNEL_CHIP}/configs/generated/chip.pb)
  set(WORKER_PROTO_DIR ${REPO_ROOT}/bbdev/api/steps/config/scripts)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${WORKER_CHIP_PB} ${WORKER_CPU_SCRIPT} ${WORKER_PROTO_DIR}/chip_pb2.py)
  execute_process(COMMAND python3 ${WORKER_CPU_SCRIPT}
    --chip-pb ${WORKER_CHIP_PB} --proto-dir ${WORKER_PROTO_DIR}
    --workers ${WORKER_COUNT} --hart-count ${BUCKYBALL_HART_COUNT}
    OUTPUT_VARIABLE WORKER_CPUS OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE WORKER_CPU_RESULT)
  if(NOT WORKER_CPU_RESULT EQUAL 0)
    message(FATAL_ERROR "Cannot resolve worker affinity from chip.pb")
  endif()
  message(STATUS "Worker CPUs in Tile order: ${WORKER_CPUS}")
  set(SERVING_ARGS --worker-cpus ${WORKER_CPUS})
  set(MODEL_INIT_TEMPLATE ${REPO_ROOT}/stack/serving/loader/guest/init.in)
  set(SERVING_QUIET_CONSOLE ON)
  if(BUCKYBALL_MODEL_STORAGE STREQUAL "ddr")
    execute_process(COMMAND python3 ${MODEL_DDR_SCRIPT} check --source ${MODEL_OUTPUT_DIR}
      --guest-memory-mib ${BUCKYBALL_GUEST_MEMORY_MIB} RESULT_VARIABLE BUDGET_RESULT)
  else()
    execute_process(COMMAND python3 ${MEMORY_BUDGET_SCRIPT}
      --resources ${MODEL_OUTPUT_DIR} --guest-memory-mib ${BUCKYBALL_GUEST_MEMORY_MIB}
      RESULT_VARIABLE BUDGET_RESULT)
  endif()
  if(NOT BUDGET_RESULT EQUAL 0)
    message(FATAL_ERROR "Model resources exceed the selected storage budget")
  endif()
  set(GUEST_SOURCE ${REPO_ROOT}/stack/serving/loader/guest)
  set(GUEST_BINARY ${CMAKE_BINARY_DIR}/guest/worker-mux)
  set(GUEST_EXTRA_TARGET)
  set(GUEST_EXTRA_BINARY)
  if(BUCKYBALL_MODEL_STORAGE STREQUAL "ddr")
    set(GUEST_EXTRA_TARGET resource-check)
    set(GUEST_EXTRA_BINARY ${CMAKE_BINARY_DIR}/guest/resource-check)
  endif()
  add_custom_command(OUTPUT ${GUEST_BINARY} ${GUEST_EXTRA_BINARY}
    COMMAND ${CMAKE_COMMAND} -S ${GUEST_SOURCE} -B ${CMAKE_BINARY_DIR}/guest
      -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=riscv64
      -DCMAKE_CXX_COMPILER=${CROSS_COMPILE}g++
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}/guest --target worker-mux ${GUEST_EXTRA_TARGET}
    DEPENDS ${GUEST_SOURCE}/CMakeLists.txt ${GUEST_SOURCE}/mux.cpp ${GUEST_SOURCE}/resource-check.cpp ${GUEST_SOURCE}/../../../runtime/include/resources.h VERBATIM)
  add_custom_target(serving-payload DEPENDS ${GUEST_BINARY} ${GUEST_EXTRA_BINARY})
  set(SERVING_DEPENDS serving-payload ${GUEST_BINARY} ${GUEST_EXTRA_BINARY} ${WORKER_CPU_SCRIPT} ${WORKER_CHIP_PB})
  set(SERVING_INSTALL_COMMAND
    COMMAND ${CMAKE_COMMAND} -E make_directory ${ROOTFS_DIR}/root/guest
    COMMAND ${CMAKE_COMMAND} -E copy ${GUEST_BINARY} ${ROOTFS_DIR}/root/guest/worker-mux)
  if(BUCKYBALL_MODEL_STORAGE STREQUAL "ddr")
    list(APPEND SERVING_INSTALL_COMMAND COMMAND ${CMAKE_COMMAND} -E copy
      ${GUEST_EXTRA_BINARY} ${ROOTFS_DIR}/root/guest/resource-check)
  endif()
  set(BUDGET_ROOTFS_COMMAND COMMAND python3 ${MEMORY_BUDGET_SCRIPT}
    --rootfs ${ROOTFS_DIR} --guest-memory-mib ${BUCKYBALL_GUEST_MEMORY_MIB})
  set(BUDGET_IMAGE_COMMAND COMMAND python3 ${MEMORY_BUDGET_SCRIPT}
    --rootfs ${ROOTFS_DIR} --image ${CMAKE_BINARY_DIR}/linux/arch/riscv/boot/Image
    --guest-memory-mib ${BUCKYBALL_GUEST_MEMORY_MIB})
elseif(NOT MODEL_EXECUTION STREQUAL "native")
  message(FATAL_ERROR "Kernel model images require native or host-worker guest execution")
endif()
if(BUCKYBALL_MODEL_STORAGE STREQUAL "ddr" AND MODEL_EXECUTION STREQUAL "native")
  execute_process(COMMAND python3 ${MODEL_DDR_SCRIPT} check --source ${MODEL_OUTPUT_DIR}
    --guest-memory-mib ${BUCKYBALL_GUEST_MEMORY_MIB} COMMAND_ERROR_IS_FATAL ANY)
  set(GUEST_SOURCE ${REPO_ROOT}/stack/serving/loader/guest)
  set(GUEST_BINARY ${CMAKE_BINARY_DIR}/guest/resource-check)
  add_custom_command(OUTPUT ${GUEST_BINARY}
    COMMAND ${CMAKE_COMMAND} -S ${GUEST_SOURCE} -B ${CMAKE_BINARY_DIR}/guest
      -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=riscv64
      -DCMAKE_CXX_COMPILER=${CROSS_COMPILE}g++
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}/guest --target resource-check
    DEPENDS ${GUEST_SOURCE}/CMakeLists.txt ${GUEST_SOURCE}/resource-check.cpp
      ${GUEST_SOURCE}/../../../runtime/include/resources.h VERBATIM)
  add_custom_target(serving-payload DEPENDS ${GUEST_BINARY})
  set(SERVING_DEPENDS serving-payload ${GUEST_BINARY})
  set(SERVING_INSTALL_COMMAND
    COMMAND ${CMAKE_COMMAND} -E make_directory ${ROOTFS_DIR}/root/guest
    COMMAND ${CMAKE_COMMAND} -E copy ${GUEST_BINARY} ${ROOTFS_DIR}/root/guest/resource-check)
endif()

foreach(output IN ITEMS SERVING_ARGS MODEL_INIT_TEMPLATE SERVING_QUIET_CONSOLE
    SERVING_DEPENDS SERVING_INSTALL_COMMAND BUDGET_ROOTFS_COMMAND BUDGET_IMAGE_COMMAND)
  set(${output} "${${output}}" PARENT_SCOPE)
endforeach()
