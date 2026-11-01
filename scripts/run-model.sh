#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <x86|nv> <model> [buddy-cli arguments]" >&2
  exit 2
fi
ARCH=$1
MODEL=$2
shift 2
case "$ARCH" in
  x86|nv) ;;
  *) echo "Unknown host backend: $ARCH" >&2; exit 2 ;;
esac
BUILD="$ROOT/output/models/$ARCH/$MODEL"
cmake -S "$ROOT/models" -B "$BUILD" -G Ninja -DARCH="$ARCH" -DMODEL="$MODEL"
cmake --build "$BUILD" --target "buddy-$ARCH-$MODEL-rax-origin-build"
exec "$ROOT/thirdparty/buddy-mlir/build/bin/buddy-cli" \
  --model "$ROOT/models/archs/$ARCH/$MODEL/origin/output/$MODEL.rax" "$@"
