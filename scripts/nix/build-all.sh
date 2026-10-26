#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
BUDDY="$ROOT/thirdparty/buddy-mlir"
ONLY_LIST=()
SKIP_LIST=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h | --help)
      cat <<'USAGE'
Usage: ./scripts/nix/build-all.sh [OPTIONS]

Steps:
  0  Build the Nix environment
  1  Initialize Buddy-MLIR and LLVM sources
  2  Build LLVM/MLIR
  3  Build Buddy-MLIR

Options:
  -h       Show this help
  -o N     Run only step N; repeat to select multiple steps
  -s N     Skip step N; repeat to skip multiple steps

Selected steps require their dependencies to have been built.
-o and -s cannot be combined.
USAGE
      exit 0 ;;
    -o | -s)
      option=$1
      shift
      if [ "$#" -eq 0 ] || [[ ! "$1" =~ ^[0-3]$ ]]; then
        echo "Error: $option requires a step number from 0 to 3" >&2
        exit 2
      fi
      case "$option" in
        -o) ONLY_LIST+=("$1") ;;
        -s) SKIP_LIST+=("$1") ;;
      esac ;;
    *)
      echo "Error: invalid option $1" >&2
      exit 2 ;;
  esac
  shift
done

if [ "${#ONLY_LIST[@]}" -gt 0 ] && [ "${#SKIP_LIST[@]}" -gt 0 ]; then
  echo "Error: -o and -s cannot be combined" >&2
  exit 2
fi

run_step() {
  local step
  if [ "${#ONLY_LIST[@]}" -gt 0 ]; then
    for step in "${ONLY_LIST[@]}"; do
      [ "$step" = "$1" ] && return 0
    done
    return 1
  fi
  for step in "${SKIP_LIST[@]}"; do
    [ "$step" = "$1" ] && return 1
  done
  return 0
}

cd "$ROOT"
if run_step 0; then
  echo "[0] Building the Nix environment"
  nix build
fi

REMAINING=()
for step in 1 2 3; do
  if run_step "$step"; then
    REMAINING+=(-o "$step")
  fi
done
if [ "${#REMAINING[@]}" -eq 0 ]; then
  exit 0
fi
if [ -z "${IN_NIX_SHELL:-}" ]; then
  exec nix develop -c bash "$ROOT/scripts/nix/build-all.sh" "${REMAINING[@]}"
fi

if run_step 1; then
  echo "[1] Initializing Git submodules"
  git submodule update --init thirdparty/buddy-mlir
  git -C "$BUDDY" submodule update --init llvm
fi

if run_step 2; then
  echo "[2] Building LLVM/MLIR"
  NVCC=$(command -v nvcc)
  CUDA_ROOT=$(dirname "$(dirname "$NVCC")")
  cmake -G Ninja -S "$BUDDY/llvm/llvm" -B "$BUDDY/llvm/build" \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV;NVPTX" \
    -DMLIR_ENABLE_CUDA_RUNNER=ON \
    -DCMAKE_CUDA_COMPILER="$NVCC" \
    -DCUDAToolkit_ROOT="$CUDA_ROOT" \
    -DCMAKE_CUDA_HOST_COMPILER="$(command -v g++-13)" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE="$(command -v python3)"
  ninja -C "$BUDDY/llvm/build"
fi

if run_step 3; then
  echo "[3] Building Buddy-MLIR"
  cmake -G Ninja -S "$BUDDY" -B "$BUDDY/build" \
    -DMLIR_DIR="$BUDDY/llvm/build/lib/cmake/mlir" \
    -DLLVM_DIR="$BUDDY/llvm/build/lib/cmake/llvm" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DPython3_EXECUTABLE="$(command -v python3)" \
    -DPython_EXECUTABLE="$(command -v python3)"
  ninja -C "$BUDDY/build"
fi
