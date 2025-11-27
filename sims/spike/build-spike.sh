#!/bin/bash

ROOT=$(git rev-parse --show-toplevel)
CYDIR=$ROOT/thirdparty/chipyard

cd "$CYDIR/generators/gemmini/spike"

mkdir -p build && cd build

cmake ..
make install

cd "$ROOT/toolchains/riscv-tools/riscv-isa-sim/build"
make
make install

echo "Spike Gemmini Extension Build completed!"
