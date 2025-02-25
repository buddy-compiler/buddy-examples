# buddy-examples

This repository provides examples of using the Buddy Compiler to run inference on large models such as DeepSeek-R1.

## Available Examples

The table below lists the supported large models:

| Name  | Build Target |
| -------------- | ------------- |
| DeepSeekR1 | `ninja deepseek-r1` |


## How to Build

1. Set the `buddy-mlir` toolchain and PYTHONPATH environment variable:
Make sure that the PYTHONPATH variable includes the directory of LLVM/MLIR python bindings and the directory of Buddy MLIR python packages.

```bash
$ cd buddy-mlir/build
$ export BUDDY_MLIR_BUILD_DIR=$PWD
$ export LLVM_MLIR_BUILD_DIR=${BUDDY_MLIR_BUILD_DIR}/../llvm/build/
$ export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

2. Build on local platform:

```bash
$ cd buddy-examples
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DBUDDY_MLIR_BUILD_DIR=${BUDDY_MLIR_BUILD_DIR} \
    -DCMAKE_CXX_COMPILER=${LLVM_MLIR_BUILD_DIR}/bin/clang++ \
    -DCMAKE_C_COMPILER=${LLVM_MLIR_BUILD_DIR}/bin/clang \
    -DCMAKE_CXX_FLAGS=-march=native \
    -DCMAKE_C_FLAGS=-march=native
$ ninja <target>
// For example: 
$ ninja deepseek-r1
$ ./bin/<target>
```

### Cross Compile to Target Platform

**RISC-V Vector Extension**

Follow the [Environment Setup Guide for MLIR and RVV Testing and Experiments](https://github.com/buddy-compiler/buddy-mlir/blob/main/docs/RVVEnvironment.md) to prepare the RVV environment. Furthermore, To enable the openmp feature on RISC-V, you also need to refer to [Prepare RISC-V OpenMP ToolChain](https://github.com/buddy-compiler/buddy-benchmark/blob/main/docs/PrepareRVOpenMP.md).

1. Set variables for the toolchain:

```bash
$ cd buddy-mlir/build
$ export BUDDY_MLIR_BUILD_DIR=$PWD
$ export LLVM_MLIR_BUILD_DIR=${BUDDY_MLIR_BUILD_DIR}/../llvm/build/
$ export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
$ export BUDDY_MLIR_BUILD_CROSS_DIR=${BUDDY_MLIR_BUILD_DIR}/../build-cross-rv
$ export RISCV_GNU_TOOLCHAIN=${BUDDY_MLIR_BUILD_DIR}/thirdparty/riscv-gnu-toolchain
$ export RISCV_OMP_SHARED=${LLVM_MLIR_BUILD_DIR}/../build-omp-shared-rv/libomp.so

```

2. Build on the target platform:

```bash
$ cd buddy-examples
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCROSS_COMPILE_RVV=ON \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=riscv \
    -DCMAKE_C_COMPILER=${LLVM_MLIR_BUILD_DIR}/bin/clang \
    -DRISCV_GNU_TOOLCHAIN=${RISCV_GNU_TOOLCHAIN} \
    -DCMAKE_CXX_COMPILER=${LLVM_MLIR_BUILD_DIR}/bin/clang++ \
    -DCMAKE_C_FLAGS="-march=rv64gcv --target=riscv64-unknown-linux-gnu --sysroot=${RISCV_GNU_TOOLCHAIN}/sysroot --gcc-toolchain=${RISCV_GNU_TOOLCHAIN} -fPIC" \
    -DCMAKE_CXX_FLAGS="-march=rv64gcv --target=riscv64-unknown-linux-gnu --sysroot=${RISCV_GNU_TOOLCHAIN}/sysroot --gcc-toolchain=${RISCV_GNU_TOOLCHAIN} -fPIC" \
    -DRISCV_OMP_SHARED=${RISCV_OMP_SHARED} \
    -DBUDDY_MLIR_BUILD_DIR=${BUDDY_MLIR_BUILD_DIR} \
    -DBUDDY_MLIR_BUILD_CROSS_DIR=${BUDDY_MLIR_BUILD_CROSS_DIR} \
    -DBUDDY_MLIR_CROSS_LIB_DIR=${BUDDY_MLIR_BUILD_CROSS_DIR}/lib

$ ninja <target>
// For example: 
$ ninja deepseek-r1
```

3. Transfer the compiled file in `build/bin/` to your target platform and run it.
