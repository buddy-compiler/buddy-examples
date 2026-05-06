#!/usr/bin/env bash

# exit script if any command fails
set -e
set -o pipefail

ROOT=$(git rev-parse --show-toplevel)

usage() {
  echo "Usage: ${0} [OPTIONS] "
  echo ""
  echo "Helper script to fully initialize repository that wraps other scripts."
  echo "By default it initializes/installs things in the following order:"
  echo "   1. bbdev xxx"
  echo "   2. Compiler installation"
  echo "   3. RTL pre-compile sources"
  echo "   4. bb-tests pre-compile sources"
  echo "   5. waveform-mcp build"
  echo "   6. pre-commit hooks installation"
  echo ""
  echo "**See below for options to skip parts of the setup. Skipping parts of the setup is not guaranteed to be tested/working.**"
  echo ""
  echo "Options"
  echo "  --help -h     : Display this message"
  echo "  --skip -s N   : Skip step N in the list above. Use multiple times to skip multiple steps ('-s N -s M ...')."
  exit "$1"
}

SKIP_LIST=()
VERBOSE_FLAG=""
INSTALL_IN_NIX=0

while [ "$1" != "" ];
do
  case $1 in
    -h | --help )
      usage 3 ;;
    --verbose | -v)
      VERBOSE_FLAG=$1
      set -x ;;
    --skip | -s)
      shift
      SKIP_LIST+=(${1}) ;;
    --install-in-nix)
      INSTALL_IN_NIX=1 ;;
    * )
      echo "Error: invalid option $1" >&2
      usage 1 ;;
  esac
  shift
done

# return true if the arg is not found in the SKIP_LIST
run_step() {
  local value=$1
  [[ ! " ${SKIP_LIST[*]} " =~ " ${value} " ]]
}

function begin_step
{
  thisStepNum=$1;
  thisStepDesc=$2;

  # Color codes
  local BLUE='\033[0;34m'
  local GREEN='\033[0;32m'
  local YELLOW='\033[1;33m'
  local NC='\033[0m' # No Color

  echo -e "${BLUE} ========================================================================="
  echo -e "${GREEN} ==== BUCKYBALL SETUP STEP ${YELLOW}$thisStepNum${GREEN}: ${YELLOW}$thisStepDesc${GREEN} "
  echo -e "${BLUE} ========================================================================="
  echo -e "${NC}"
}

begin_step "1" "submodules init and nix environment setup"
cd ${ROOT}
# git submodule update --init
nix build
if [ "${INSTALL_IN_NIX}" != "1" ]; then
  SKIP_ARGS=""
  for skip in "${SKIP_LIST[@]}"; do
    SKIP_ARGS="${SKIP_ARGS} -s ${skip}"
  done
  exec nix develop --command bash ${ROOT}/scripts/nix/build-all.sh --install-in-nix ${SKIP_ARGS} ${VERBOSE_FLAG}
fi

# if run_step "2"; then
#   begin_step "2" "Compiler installation"
#   cd ${ROOT}/compiler/thirdparty/buddy-mlir
#   git submodule update --init llvm

#   mkdir -p llvm/build && cd llvm/build
#   cmake -G Ninja ../llvm \
#     -DLLVM_ENABLE_PROJECTS="mlir;clang" \
#     -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
#     -DLLVM_ENABLE_ASSERTIONS=ON \
#     -DCMAKE_BUILD_TYPE=RELEASE \
#     -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
#     -DPython3_EXECUTABLE=$(which python3)
#   ninja #check-mlir check-clang

#   cd ${ROOT}/compiler/thirdparty/buddy-mlir
#   mkdir -p build && cd build
#   cmake -G Ninja .. \
#     -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
#     -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
#     -DLLVM_ENABLE_ASSERTIONS=ON \
#     -DCMAKE_BUILD_TYPE=RELEASE \
#     -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
#     -DPython3_EXECUTABLE=$(which python3) \
#     -DPython_EXECUTABLE=$(which python3) \
#     -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
#   ninja # check-buddy
# fi

# if run_step "3"; then
#   begin_step "3" "arch pre-compile sources"
#   # Generate firrtl2 ANTLR and compile firrtl2 in chipyard first (avoids antlr missing when arch compiles chipyard)
#   cd ${ROOT}/arch/thirdparty/chipyard
#   sbt -J-Xms512m -J-Xmx4g -J-XX:+UseG1GC "firrtl2/compile"
#   cd ${ROOT}/arch
#   bbdev verilator --verilog '--config sims.verilator.BuckyballToyVerilatorConfig'
# fi

begin_step "END" "Setup completed successfully!"
