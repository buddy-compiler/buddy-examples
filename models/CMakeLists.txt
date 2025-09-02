set(MODEL_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(BUDDY_MLIR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/buddy-mlir)
set(RISCV_GNU_TOOLCHAIN $ENV{RISCV})

add_subdirectory(LeNet)
add_subdirectory(DeepSeekR1)
