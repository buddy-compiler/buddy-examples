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
#   RUNTIME_PARAMS    ISA directory whose params.h sizes the DMA helpers; a
#                     device without banks has none and no DMA helpers
# ===----------------------------------------------------------------------===
set(RUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(MLIR_EXECUTION_ENGINE "${RUNTIME_DIR}/../thirdparty/buddy-mlir/llvm/mlir/include/mlir/ExecutionEngine")

# ===----------------------------------------------------------------------===
# Job scheduling over cores. Freestanding, so baremetal and Linux programs
# both link it.
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

add_library(runtime-shared STATIC scheduler/runtime.c)
target_include_directories(runtime-shared
  PUBLIC "$<BUILD_INTERFACE:${RUNTIME_DIR}/scheduler>"
)
target_compile_features(runtime-shared PUBLIC c_std_11)
target_compile_options(runtime-shared PRIVATE -O2 -ffreestanding -fno-builtin -msmall-data-limit=0)
target_link_libraries(runtime-shared
  INTERFACE bindings
            engine-shared
)

# ===----------------------------------------------------------------------===
# One worker per core on baremetal harts.
# ===----------------------------------------------------------------------===
add_library(runtime-baremetal STATIC scheduler/baremetal.c)
target_link_libraries(runtime-baremetal
  PUBLIC runtime
)
target_compile_options(runtime-baremetal PRIVATE -O2 -ffreestanding -fno-builtin -msmall-data-limit=0)

# ===----------------------------------------------------------------------===
# One pinned pthread worker per core on Linux, plus the services compiled
# programs call: workspace, DMA, core links, MLIR runner utilities and traces.
# ===----------------------------------------------------------------------===
set(
  RUNTIME_LINUX_SOURCES
  scheduler/linux.c
  services/workspace.c
  services/corelink.c
  services/mlir.cpp
  trace/trace.cpp
  trace/location.cpp
)
if(RUNTIME_PARAMS)
  list(APPEND RUNTIME_LINUX_SOURCES services/dma.c)
endif()

add_library(runtime-linux STATIC ${RUNTIME_LINUX_SOURCES})
target_include_directories(runtime-linux
  PUBLIC "$<BUILD_INTERFACE:${RUNTIME_DIR}/services>"
         PRIVATE
         "${RUNTIME_PARAMS}"
         "${RUNTIME_DIR}/trace"
         "${MLIR_EXECUTION_ENGINE}"
)
target_compile_options(runtime-linux PRIVATE -O2 -ftls-model=local-exec)
target_compile_features(runtime-linux PRIVATE c_std_11 cxx_std_17)
# Traces name the hardware thread of the current instruction stream.
target_link_libraries(runtime-linux
  PUBLIC runtime
         engine
         pthread
)

add_library(runtime-linux-shared STATIC ${RUNTIME_LINUX_SOURCES})
target_include_directories(runtime-linux-shared
  PUBLIC "$<BUILD_INTERFACE:${RUNTIME_DIR}/services>"
         PRIVATE
         "${RUNTIME_PARAMS}"
         "${RUNTIME_DIR}/trace"
         "${MLIR_EXECUTION_ENGINE}"
)
target_compile_options(runtime-linux-shared PRIVATE -O2)
target_compile_features(runtime-linux-shared PRIVATE c_std_11 cxx_std_17)
target_link_libraries(runtime-linux-shared
  PUBLIC runtime-shared
         engine-shared
         pthread
)
