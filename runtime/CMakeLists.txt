get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
add_library(runtime_ant STATIC src/ant.c src/tlink.c)
target_include_directories(runtime_ant PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(runtime_ant PRIVATE -O2)
add_library(runtime STATIC src/runtime.c src/workspace.c src/mlir.cpp src/trace.cpp src/trace_location.cpp)
target_compile_options(runtime PRIVATE -O2)
target_include_directories(runtime
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${RUNTIME_PARAMS}
          ${REPO_ROOT}/stack/compiler/thirdparty/buddy-mlir/llvm/mlir/include/mlir/ExecutionEngine)
target_include_directories(runtime PRIVATE ${REPO_ROOT}/bb-tests/workloads/lib)
target_compile_features(runtime PRIVATE c_std_11 cxx_std_17)

add_library(runtime_linux STATIC EXCLUDE_FROM_ALL src/linux_tasks.c src/workspace.c src/dma.c src/mlir.cpp src/trace.cpp src/trace_location.cpp)
target_compile_options(runtime_linux PRIVATE -O2)
get_target_property(RUNTIME_INCLUDES runtime INCLUDE_DIRECTORIES)
target_include_directories(runtime_linux PUBLIC ${RUNTIME_INCLUDES})
target_include_directories(runtime_linux PRIVATE ${ISA_DIR}/chip)
target_compile_features(runtime_linux PRIVATE c_std_11 cxx_std_17)
target_link_libraries(runtime_linux PUBLIC pthread)

# Controller glue uses this backend explicitly; the Ant image has only local LSU access.
set(ANT_STREAM_DIR "${CMAKE_CURRENT_BINARY_DIR}/ant-stream")
file(MAKE_DIRECTORY "${ANT_STREAM_DIR}")
set(ANT_STREAM_CASES "")
foreach(OP RANGE 0 127)
  string(APPEND ANT_STREAM_CASES "case ${OP}: __asm__ volatile(\".insn r 0x7b, 3, ${OP}, x0, %0, %1\" :: \"r\"(op->rs1), \"r\"(op->rs2) : \"memory\"); break;\n")
endforeach()
file(GENERATE OUTPUT "${ANT_STREAM_DIR}/instructions.inc" CONTENT "${ANT_STREAM_CASES}")
add_custom_command(
  OUTPUT "${ANT_STREAM_DIR}/stream.elf" "${ANT_STREAM_DIR}/stream.bin"
  COMMAND "$ENV{RISCV}/bin/riscv64-unknown-elf-gcc" -O2 -march=rv64im -mabi=lp64
    -mcmodel=medany -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
    -fno-jump-tables -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
    -nostdlib -nostartfiles -Wl,--no-relax -Wl,-T,${CMAKE_CURRENT_SOURCE_DIR}/src/ant/code.ld
    -I${ANT_STREAM_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src/ant/start.S
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ant/stream.c -o "${ANT_STREAM_DIR}/stream.elf"
  COMMAND "$ENV{RISCV}/bin/riscv64-unknown-elf-objcopy" -O binary --only-section=.text
    "${ANT_STREAM_DIR}/stream.elf" "${ANT_STREAM_DIR}/stream.bin"
  DEPENDS src/ant/start.S src/ant/stream.c src/ant/code.ld "${ANT_STREAM_DIR}/instructions.inc")
enable_language(ASM)
file(GENERATE OUTPUT "${ANT_STREAM_DIR}/image.S" CONTENT
  ".section .rodata.ant,\"a\",@progbits\n.balign 8\n.global ant_stream_image, ant_stream_image_end\nant_stream_image:\n.incbin \"${ANT_STREAM_DIR}/stream.bin\"\nant_stream_image_end:\n.section .note.GNU-stack,\"\",@progbits\n")
set_source_files_properties("${ANT_STREAM_DIR}/image.S" PROPERTIES OBJECT_DEPENDS "${ANT_STREAM_DIR}/stream.bin")
add_library(runtime_ant_tasks STATIC EXCLUDE_FROM_ALL src/ant_tasks.c src/workspace.c src/dma.c src/mlir.cpp src/trace.cpp src/trace_location.cpp "${ANT_STREAM_DIR}/image.S")
get_target_property(RUNTIME_INCLUDES runtime INCLUDE_DIRECTORIES)
target_include_directories(runtime_ant_tasks PUBLIC ${RUNTIME_INCLUDES})
target_compile_definitions(runtime_ant_tasks PUBLIC BUCKYBALL_ANT_HOST=1)
target_compile_options(runtime_ant_tasks PRIVATE -O2)
target_compile_features(runtime_ant_tasks PRIVATE c_std_11 cxx_std_17)
target_link_libraries(runtime_ant_tasks PUBLIC runtime_ant pthread)
