# buddy-examples

This repository provides examples of using the Buddy Compiler to run inference on large models such as DeepSeek-R1.

## Available Examples

The table below lists the supported models:

| Name  | Build Target |
| -------------- | ------------- |
| DeepSeekR1 | `ninja deepseek-r1` |


## How to Build on x86

1. Set the `buddy-mlir` toolchain and PYTHONPATH environment variable:

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
$ ninja <target> # For example: `ninja deepseek-r1`
$ ./bin/<target> # Directly run the target
```

## Cross Compile to RVV Platform

### Environmental Setup

1. Follow the [Environment Setup Guide for MLIR and RVV Testing and Experiments](https://github.com/buddy-compiler/buddy-mlir/blob/main/docs/RVVEnvironment.md) to prepare the RVV environment. 

2. Since the repository depends on OpenMP shared libraries, follow the steps below to set up the OpenMP dependency:

```bash
$ cd ${LLVM_MLIR_BUILD_DIR}/../
$ wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1XEsAhOcMioN9gdufuyO9OrHIdR0UtHh2' -O build-omp-shared-rv.tar.gz
$ mkdir build-omp-shared-rv && tar -xzf build-omp-shared-rv.tar.gz -C build-omp-shared-rv && rm build-omp-shared-rv.tar.gz
```

### How to Build
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
$ ninja <target> # For example: `ninja deepseek-r1`
```

3. Transfer `build/bin/` directory to your RVV platform, then run `build/bin/<target>`.
