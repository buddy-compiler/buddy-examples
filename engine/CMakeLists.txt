# ===----------------------------------------------------------------------===
# The engine: the OS-invisible control code a core runs to schedule jobs onto
# its hardware threads, and the instruction streams host code issues to them.
#
# `engine` is linked by the device; the loaders of its hardware threads
# register themselves in `engine_bindings`. On baremetal the engine issues the
# core's management instructions itself (src/thread.c); under Linux the OS
# owns the threads and the engine driver (stack/os/linux/driver) issues them
# (src/thread_linux.c).
#   engine         baremetal programs; they have no __tls_get_addr, so
#                  thread-local variables use local-exec
#   engine-linux   statically linked Linux programs
#   engine-shared  shared objects Linux loads at run time, such as models
# ===----------------------------------------------------------------------===
set(DRIVER "${CMAKE_CURRENT_SOURCE_DIR}/../os/linux/driver")

add_library(engine STATIC src/engine.c src/thread.c src/stream.c src/location.c)
target_include_directories(engine
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)
target_compile_features(engine PUBLIC c_std_11)
target_compile_options(
  engine
  PRIVATE -O2 -ffreestanding -fno-builtin -msmall-data-limit=0 -ftls-model=local-exec
)

add_library(engine-linux STATIC src/engine.c src/thread_linux.c src/stream.c src/location.c)
target_include_directories(engine-linux
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
         PRIVATE
         "${DRIVER}"
)
target_compile_definitions(engine-linux
  PRIVATE ENGINE_LINUX
)
target_compile_features(engine-linux PUBLIC c_std_11)
target_compile_options(engine-linux PRIVATE -O2 -ftls-model=local-exec)

add_library(engine-shared STATIC src/engine.c src/thread_linux.c src/stream.c src/location.c)
target_include_directories(engine-shared
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
         PRIVATE
         "${DRIVER}"
)
target_compile_definitions(engine-shared
  PRIVATE ENGINE_LINUX
)
target_compile_features(engine-shared PUBLIC c_std_11)
target_compile_options(engine-shared PRIVATE -O2)
