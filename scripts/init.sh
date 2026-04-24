#!/usr/bin/env bash

# exit script if any command fails
set -e
set -o pipefail

ROOT=$(git rev-parse --show-toplevel)

source ${ROOT}/scripts/utils.sh

usage() {
  echo "Usage: ${0} [OPTIONS] "
  echo ""
  echo "Helper script to fully initialize repository that wraps other scripts."
  echo "By default it initializes/installs things in the following order:"
  echo "   0. init env.sh"
  echo "   1. init submodules"
  echo "   2. Chipyard environment setup"
  echo "   3. Buddy-mlir pre-compile sources"
  echo ""
  echo "**See below for options to skip parts of the setup. Skipping parts of the setup is not guaranteed to be tested/working.**"
  echo ""
  echo "Options"
  echo "  --help -h   : Display this message"
  echo "  --verbose -v  : Verbose printout"
  echo "  --skip -s N   : Skip step N in the list above. Use multiple times to skip multiple steps ('-s N -s M ...')."
  echo "  --admin     : Add this option to install the admin tools (You dont need do this)."
  echo "  --conda-env-name <name> : Add this option to specify the conda environment name. Default is buddy-mlir."

  exit "$1"
}

SKIP_LIST=()
VERBOSE_FLAG=""
ADMIN_MODE=false
CONDA_ENV_NAME="buddy"

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
  --admin)
    ADMIN_MODE=true ;;
  --conda-env-name)
    shift
    CONDA_ENV_NAME=${1} ;;
  * )
    error "invalid option $1"
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
  local NC='\033[0m' 
  
  echo -e "${BLUE} ========================================================================="
  echo -e "${GREEN} ==== BUDDY-EXAMPLE SETUP STEP ${YELLOW}$thisStepNum${GREEN}: ${YELLOW}$thisStepDesc${GREEN} "
  echo -e "${BLUE} ========================================================================="
  echo -e "${NC}"
}

if run_step "0"; then
  begin_step "0" "init env.sh"
  replace_content ${ROOT}/env.sh base-conda-setup "source $(conda info --base)/etc/profile.d/conda.sh"
fi

if run_step "1"; then
  begin_step "1" "submodules init"
  git submodule update --init 
  replace_content ${ROOT}/thirdparty/chipyard/env.sh base-conda-setup "source $(conda info --base)/etc/profile.d/conda.sh"
fi

# setup and install chipyard environment
if run_step "2"; then
  begin_step "2" "Chipyard environment setup"
  cd ${ROOT}/thirdparty/chipyard && ./build-setup.sh --conda-env-name ${CONDA_ENV_NAME}
  cp ${ROOT}/thirdparty/chipyard/env.sh ${ROOT}/env.sh
  replace_content ${ROOT}/env.sh build-setup-conda "conda activate ${CONDA_ENV_NAME}
source ${ROOT}/thirdparty/chipyard/scripts/fix-open-files.sh"
  replace_content ${ROOT}/env.sh bb-dir-helper "ROOT=${ROOT}"
fi

if run_step "3"; then
  begin_step "3" "Compiler (buddy-mlir) pre-compile sources"
  cd ${ROOT}
  source ${ROOT}/env.sh
  ./scripts/install-buddy-compiler.sh
fi
