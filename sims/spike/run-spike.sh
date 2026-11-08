#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <Spike arguments and ELF>" >&2
  exit 2
fi
exec "$ROOT/output/spike/bin/spike" \
  --extlib="$ROOT/output/spike/lib/libgemmini.so" --extension=gemmini "$@"
