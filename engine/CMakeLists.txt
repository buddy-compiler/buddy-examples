# ===----------------------------------------------------------------------===
# The engine: the OS-invisible control code a core runs to schedule jobs onto
# its hardware threads, and the instruction streams host code issues to them.
#
# `engine` is linked by the device; the loaders of its hardware threads
# register themselves in `engine_bindings`.
#   engine         for statically linked programs; baremetal ones have no
#                  __tls_get_addr, so thread-local variables use local-exec
#   engine-shared  for shared objects Linux loads at run time, such as models
# ===----------------------------------------------------------------------===
set(ENGINE_SOURCES src/engine.c src/thread.c src/stream.c)

add_library(engine STATIC ${ENGINE_SOURCES})
target_include_directories(engine
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)
target_compile_features(engine PUBLIC c_std_11)
target_compile_options(
  engine
  PRIVATE -O2 -ffreestanding -fno-builtin -msmall-data-limit=0 -ftls-model=local-exec
)

add_library(engine-shared STATIC ${ENGINE_SOURCES})
target_include_directories(engine-shared
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)
target_compile_features(engine-shared PUBLIC c_std_11)
target_compile_options(engine-shared PRIVATE -O2 -ffreestanding -fno-builtin -msmall-data-limit=0)
