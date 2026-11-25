# ===----------------------------------------------------------------------===
# The kernel: the OS-invisible control code a hart runs to schedule jobs onto
# its hardware threads, and the instruction streams host code issues to them.
#
# `kernel` is linked by the device; the loaders of its hardware threads
# register themselves in `kernel_bindings`. On baremetal the kernel issues the
# hart's management instructions itself (src/thread.c); under Linux the OS
# owns the threads and the kernel driver (stack/os/linux/driver) issues them
# (src/thread_linux.c).
#   kernel         baremetal programs; they have no __tls_get_addr, so
#                  thread-local variables use local-exec
#   kernel-linux   statically linked Linux programs
#   kernel-shared  shared objects Linux loads at run time, such as models
# ===----------------------------------------------------------------------===
set(DRIVER "${CMAKE_CURRENT_SOURCE_DIR}/../os/linux/driver")

add_library(kernel STATIC src/kernel.c src/thread.c src/stream.c src/location.c)
target_include_directories(kernel
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)
target_compile_features(kernel PUBLIC c_std_11)
target_compile_options(
  kernel
  PRIVATE -O2 -ffreestanding -fno-builtin -msmall-data-limit=0 -ftls-model=local-exec
)

add_library(kernel-linux STATIC src/kernel.c src/thread_linux.c src/stream.c src/location.c)
target_include_directories(kernel-linux
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
         PRIVATE
         "${DRIVER}"
)
target_compile_definitions(kernel-linux
  PRIVATE KERNEL_LINUX
)
target_compile_features(kernel-linux PUBLIC c_std_11)
target_compile_options(kernel-linux PRIVATE -O2 -ftls-model=local-exec)

add_library(kernel-shared STATIC src/kernel.c src/thread_linux.c src/stream.c src/location.c)
target_include_directories(kernel-shared
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
         PRIVATE
         "${DRIVER}"
)
target_compile_definitions(kernel-shared
  PRIVATE KERNEL_LINUX
)
target_compile_features(kernel-shared PUBLIC c_std_11)
target_compile_options(kernel-shared PRIVATE -O2)
