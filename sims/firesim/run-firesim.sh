#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
CONFIG="$ROOT/sims/firesim/yaml"
cd "$ROOT/platforms/thirdparty/chipyard/sims/firesim/deploy"
./firesim infrasetup \
  -a "$CONFIG/config_hwdb.yaml" \
  -b "$CONFIG/config_build.yaml" \
  -r "$CONFIG/config_build_recipes.yaml" \
  -c "$CONFIG/config_runtime.yaml"
./firesim runworkload \
  -a "$CONFIG/config_hwdb.yaml" \
  -b "$CONFIG/config_build.yaml" \
  -r "$CONFIG/config_build_recipes.yaml" \
  -c "$CONFIG/config_runtime.yaml"
