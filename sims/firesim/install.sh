#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
CHIPYARD="$ROOT/platforms/thirdparty/chipyard"
cd "$ROOT"
git submodule update --init platforms/thirdparty/chipyard
cd "$CHIPYARD"
./scripts/init-submodules-no-riscv-tools.sh --gemmini --nvdla --saturn --cva6 --radiance
git submodule update --init sims/firesim software/firemarshal
git -C sims/firesim submodule update --init \
  sim/rocket-chip sim/berkeley-hardfloat sim/diplomacy sim/cde

# Nix provides the tools; let Chipyard create its own env.sh.
./build-setup.sh --skip-conda --skip-submodules --skip-toolchain --skip-ctags \
  -s 5 --skip-firesim --skip-marshal --skip-circt --skip-clean
