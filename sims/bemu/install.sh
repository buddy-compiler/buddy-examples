#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if [ -d "$ROOT/../bbdev" ]; then
  BEBOP="$ROOT/../bebop"
else
  git -C "$ROOT" submodule update --init thirdparty/bebop
  BEBOP="$ROOT/thirdparty/bebop"
fi
cd "$BEBOP"
exec nix run .#install
