# ===----------------------------------------------------------------------===
# The runtime: schedules jobs onto the cores of a device. Linked by the device.
#
# Statically linked programs use local-exec thread-local variables: baremetal
# programs have no __tls_get_addr. Shared objects Linux loads at run time, such
# as models, link the `-shared` variants, built with the default TLS model.
#
# Inputs, set by the device stack:
#   bindings          target the device defines: how each of its cores runs jobs
#                     (`runtime_bindings`) and each hardware thread is loaded
#                     (`engine_bindings`)
#   RUNTIME_DEFS      compile definitions of the target that sizes the DMA
#                     helpers; a device without banks has none and no DMA helpers
# ===----------------------------------------------------------------------===
set(RUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(MLIR_EXECUTION_ENGINE "${RUNTIME_DIR}/../thirdparty/buddy-mlir/llvm/mlir/include/mlir/ExecutionEngine")

# ===----------------------------------------------------------------------===
# Job scheduling over cores on baremetal harts, with the baremetal engine. The
# Linux runtimes below compile the same scheduler with the Linux engine.
# ===----------------------------------------------------------------------===
add_library(runtime STATIC scheduler/runtime.c)
target_include_directories(runtime
  PUBLIC "$<BUILD_INTERFACE:${RUNTIME_DIR}/scheduler>"
)
target_compile_features(runtime PUBLIC c_std_11)
target_compile_options(runtime PRIVATE -O2 -ffreestanding -fno-builtin -msmall-data-limit=0)
# The bindings call into the engine, so it follows them on the link line.
target_link_libraries(runtime
  INTERFACE bindings
            engine
)

# ===----------------------------------------------------------------------===
# One worker per core on baremetal harts.
# ===----------------------------------------------------------------------===
add_library(runtime-baremetal STATIC scheduler/baremetal.c services/dma_memory.c)
if(RUNTIME_DEFS)
  find_program(ELF_CC riscv64-unknown-elf-gcc REQUIRED)
  list(TRANSFORM RUNTIME_DEFS PREPEND "-D" OUTPUT_VARIABLE DMA_BAREMETAL_DEFINES)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/dma-baremetal.o
    COMMAND ${ELF_CC} -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -ftls-model=local-exec -ffreestanding -fno-builtin -msmall-data-limit=0 ${DMA_BAREMETAL_DEFINES} -I${RUNTIME_DIR}/services -c ${RUNTIME_DIR}/services/dma.c -o ${CMAKE_CURRENT_BINARY_DIR}/dma-baremetal.o
    DEPENDS services/dma.c
            services/dma.h
            services/dma_memory.h
    VERBATIM
  )
  target_sources(runtime-baremetal PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/dma-baremetal.o)
endif()
target_link_libraries(runtime-baremetal
  PUBLIC runtime
)
target_compile_options(runtime-baremetal PRIVATE -O2 -ffreestanding -fno-builtin -msmall-data-limit=0)

# ===----------------------------------------------------------------------===
# The device program runtime on baremetal harts: the program on the control
# hart submits the jobs of every core of design.pb. Built for a device that
# links it.
# ===----------------------------------------------------------------------===
list(TRANSFORM DESIGN_DEFS PREPEND "-D" OUTPUT_VARIABLE PROGRAM_DEFINES)
add_library(runtime-program STATIC EXCLUDE_FROM_ALL program/program.c)
target_include_directories(runtime-program
  PUBLIC "$<BUILD_INTERFACE:${RUNTIME_DIR}/program>"
)
target_compile_options(
  runtime-program
  PRIVATE
    -O2
    -ffreestanding
    -msmall-data-limit=0
    ${PROGRAM_DEFINES}
    "-DDEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART}"
)
target_link_libraries(runtime-program
  PUBLIC runtime-baremetal
)

# ===----------------------------------------------------------------------===
# One pinned pthread worker per core on Linux, plus the services compiled
# programs call: DMA, MLIR runner utilities and traces. A core with threads
# reaches them through the OS: its jobs link the Linux engine, whose
# management instructions the engine driver issues.
# ===----------------------------------------------------------------------===
set(
  RUNTIME_LINUX_SOURCES
  scheduler/runtime.c
  scheduler/linux.c
  services/mlir.cpp
  services/dma_memory.c
  trace/trace.cpp
  trace/location.cpp
)
if(RUNTIME_DEFS)
  list(APPEND RUNTIME_LINUX_SOURCES services/dma.c)
endif()

add_library(runtime-linux STATIC ${RUNTIME_LINUX_SOURCES})
target_include_directories(runtime-linux
  PUBLIC "$<BUILD_INTERFACE:${RUNTIME_DIR}/services>"
         "$<BUILD_INTERFACE:${RUNTIME_DIR}/scheduler>"
         PRIVATE
         "${RUNTIME_DIR}/trace"
         "${MLIR_EXECUTION_ENGINE}"
)
target_compile_definitions(runtime-linux
  PRIVATE ${RUNTIME_DEFS}
          BUCKYBALL_DEVICE_TRACE
)
target_compile_options(runtime-linux PRIVATE -O2 -ftls-model=local-exec)
target_compile_features(runtime-linux PRIVATE c_std_11 cxx_std_17)
# Traces name the hardware thread of the current instruction stream.
target_link_libraries(runtime-linux
  PUBLIC bindings
         engine-linux
         pthread
)

add_library(runtime-linux-shared STATIC ${RUNTIME_LINUX_SOURCES})
target_include_directories(runtime-linux-shared
  PUBLIC "$<BUILD_INTERFACE:${RUNTIME_DIR}/services>"
         "$<BUILD_INTERFACE:${RUNTIME_DIR}/scheduler>"
         PRIVATE
         "${RUNTIME_DIR}/trace"
         "${MLIR_EXECUTION_ENGINE}"
)
target_compile_definitions(runtime-linux-shared
  PRIVATE ${RUNTIME_DEFS}
          BUCKYBALL_DEVICE_TRACE
)
target_compile_options(runtime-linux-shared PRIVATE -O2)
target_compile_features(runtime-linux-shared PRIVATE c_std_11 cxx_std_17)
target_link_libraries(runtime-linux-shared
  PUBLIC bindings
         engine-shared
         pthread
)

add_library(runtime-program-linux STATIC EXCLUDE_FROM_ALL program/program.c)
target_include_directories(runtime-program-linux
  PUBLIC "$<BUILD_INTERFACE:${RUNTIME_DIR}/program>"
)
target_compile_definitions(runtime-program-linux
  PRIVATE PROGRAM_LINUX
          "DEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART}"
)
target_compile_options(runtime-program-linux PRIVATE -O2 ${PROGRAM_DEFINES})
target_link_libraries(runtime-program-linux
  PUBLIC runtime-linux-shared
)
