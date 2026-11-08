#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <target-config> <platform> <platform-config>" >&2
  exit 2
fi
exec make -C "$ROOT/platforms/thirdparty/chipyard/sims/firesim/sim" verilog \
  FIRESIM_ENV_SOURCED=1 TARGET_PROJECT=firesim \
  SBT="$JAVA_HOME/bin/java -Xmx8G -jar $ROOT/platforms/thirdparty/chipyard/scripts/sbt-launch.jar" \
  TARGET_PROJECT_MAKEFRAG="$ROOT/sims/firesim/makefrag/firesim" \
  PLATFORM_CONFIG_PACKAGE=firesim.configs \
  TARGET_CONFIG="$1" PLATFORM="$2" PLATFORM_CONFIG="$3"
