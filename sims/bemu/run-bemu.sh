#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <design> <device-config> <workload-or-firmware.elf>" >&2
  exit 2
fi
ELF=$(realpath "$3")
if [ -d "$ROOT/../bbdev" ]; then
  BEBOP="$ROOT/../bebop"
else
  BEBOP="$ROOT/thirdparty/bebop"
fi
cd "$BEBOP"
nix develop .#bemu -c bin/bebop-bemu config --design "$1" --device "$2"
nix develop .#bemu -c bin/bebop-bemu build --design "$1" --device "$2"
exec nix develop .#bemu -c bin/bebop-bemu run --design "$1" --device "$2" --elf "$ELF"
