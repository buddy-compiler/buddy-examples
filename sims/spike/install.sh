#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
CHIPYARD="$ROOT/platforms/thirdparty/chipyard"
PREFIX="$ROOT/output/spike"

cd "$ROOT"
git submodule update --init platforms/thirdparty/chipyard
git -C "$CHIPYARD" submodule update --init generators/gemmini toolchains/riscv-tools/riscv-isa-sim toolchains/riscv-tools/riscv-pk
git -C "$CHIPYARD/generators/gemmini" submodule update --init software/libgemmini

mkdir -p "$PREFIX/build/spike" "$PREFIX/build/pk"
cd "$PREFIX/build/spike"
"$CHIPYARD/toolchains/riscv-tools/riscv-isa-sim/configure" --prefix="$PREFIX" \
  --with-boost=no --with-boost-asio=no --with-boost-regex=no
make -j "$(nproc)"
make install
make -C "$CHIPYARD/generators/gemmini/software/libgemmini" RISCV="$PREFIX" install

cd "$PREFIX/build/pk"
"$CHIPYARD/toolchains/riscv-tools/riscv-pk/configure" --prefix="$PREFIX" \
  --host=riscv64-unknown-linux-gnu --with-arch=rv64gc_zifencei --with-abi=lp64d \
  CC="riscv64-unknown-linux-gnu-gcc -fno-tree-loop-distribute-patterns" \
  CXX=riscv64-unknown-linux-gnu-g++ \
  OBJCOPY=riscv64-unknown-linux-gnu-objcopy READELF=riscv64-unknown-linux-gnu-readelf
# Avoid the host toolchain's libutil taking precedence over pk's freestanding library.
make -j "$(nproc)" LIBS="./libutil.a -lgcc"
make LIBS="./libutil.a -lgcc" install
