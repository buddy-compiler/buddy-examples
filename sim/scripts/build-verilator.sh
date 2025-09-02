#!/bin/bash

help() {
  echo "Usage: $0 [options]"
  echo
  echo "Options:"
  echo "  -h, --help       Show this help message and exit"
  echo "  --debug        Enable debug mode"
  echo "  --fst        Enable FST waveform"
  echo "  -j <num>       Specify number of parallel jobs (default: number of cores)"
  echo "  -c, --config <config> Specify configuration parameter"
  echo "  -p, --project <project> Specify SBT project (default: chipyard)"
  echo "  -s, --sub-project <sub> Specify sub-project "
  exit 0
}

show_help=0
debug=""
j=$(nproc)
SBT_PROJECT="chipyard"
SUB_PROJECT=""

ROOT=$(git rev-parse --show-toplevel)
CONFIG=
USE_FST=
while [ $# -gt 0 ] ; do
  case $1 in
  -h|--help)
    show_help=1
    ;;
  --debug)
    debug="debug"
    ;;
  --fst)
    USE_FST=1
    ;;
  -j)
    if [[ -n $2 && $2 != -* ]]; then
    j="$2"
    shift
    else
    echo "Error: -j option requires a parameter"
    help
    fi
    ;;
  -c|--config)
    if [[ -n $2 && $2 != -* ]]; then
    CONFIG="$2"
    shift
    else
    echo "Error: -c or --config option requires a parameter"
    help
    fi
    ;;
  -p|--project)
    if [[ -n $2 && $2 != -* ]]; then
    SBT_PROJECT="$2"
    shift
    else
    echo "Error: -p or --project option requires a parameter"
    help
    fi
    ;;
  -s|--sub-project)
    if [[ -n $2 && $2 != -* ]]; then
    SUB_PROJECT="$2"
    shift
    else
    echo "Error: -s or --sub-project option requires a parameter"
    help
    fi
    ;;
  *)
    echo "Unknown option: $1"
    help
    ;;
  esac
  shift
done


if [ "$show_help" -eq 1 ]; then
  help
fi

if [ -z "$CONFIG" ]; then
  echo "ERROR: CONFIG parameter not specified. Please use -c or --config option to provide configuration."
  help
fi

CACHE_DIR="${ROOT}/.classpath_cache"

DEBUG_POSTFIX=""
if [ "$debug" == "debug" ]; then
  DEBUG_POSTFIX="-debug"
fi

cd ${ROOT}/sims/verilator/ || { echo "Cannot enter the directory: ${ROOT}/sims/verilator/"; exit 1; }
make -j$j ${debug} CONFIG=$CONFIG \
  USE_FST=$USE_FST \
  SBT_PROJECT=$SBT_PROJECT \
  $([ -n "$SUB_PROJECT" ] && echo "SUB_PROJECT=$SUB_PROJECT") \
  || { echo "[Build verilator Failed!]==================="; exit 1; }

mkdir -p ${ROOT}/sim/output/verilator
cp ${ROOT}/sims/verilator/simulator-chipyard.harness-${CONFIG}${DEBUG_POSTFIX} ${ROOT}/sim/output/verilator
