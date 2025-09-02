#!/bin/bash

help () {
  echo "Run a RISCV program on Spike, our functional ISA simulator"
  echo
  echo "Usage: $0 [-h|--help] [--pk] [--ext=EXTENSION] [--debug] BINARY"
  echo
  echo "Options:"
  echo " pk       Run binaries on the proxy kernel, which enables virtual memory"
  echo "        and a few syscalls. If this option is not set, binaries will be"
  echo "        run in baremetal mode."
  echo " ext      Specify extension to use: gemmini"
  echo " debug    Output disassembly and commit logs to log directory"
  echo " BINARY     The RISCV binary that you want to run. This can either be the"
  echo '        name of a program in `software/gemmini-rocc-tests`, or it can'
  echo "        be the full path to a binary you compiled."
  echo
  echo "Examples:"
  echo "     $0 resnet50"
  echo "     $0 --pk mvin_mvout"
  echo "     $0 --ext=gemmini path/to/binary-baremetal"
  echo "     $0 --debug --ext=gemmini path/to/binary-baremetal"
  echo "     $0 path/to/binary-baremetal"
  echo
  echo 'Note:  Run this command after running `scripts/build-spike.sh`.'
  echo
  echo "Note:  On Spike, cycle counts, SoC counter values, and performance"
  echo "     statistics are all meaningless. Use Spike only to check if your"
  echo "     programs are functionally correct. For meaningful metrics, you"
  echo "     must run your programs on VCS, Verilator, or Firesim instead."
  exit
}

if [ $# -le 0 ]; then
  help
fi

ROOT=$(git rev-parse --show-toplevel)
SIM_DIR=${ROOT}/sim
TIMESTAMP=$(date +%Y-%m-%d-%H-%M)

pk=0
show_help=0
debug=0
binary=""
extension=""

while [ $# -gt 0 ] ; do
  case $1 in
  --pk) pk=1 ;;
  --pk=*) pk="${1#--pk=}" ;;
  --ext=*) extension="${1#--ext=}" ;;
  --ext) 
    shift
    extension="$1" ;;
  --debug) debug=1 ;;
  -h | --help) show_help=1 ;;
  *) binary=$1
  esac

  shift
done

if [ $show_help -eq 1 ]; then
   help
fi

# Validate extension (if provided)
if [ -n "$extension" ] && [ "$extension" != "gemmini" ]; then
  echo "Error: Unknown extension '$extension'. Use 'gemmini'."
  exit 1
fi

if [ $pk -eq 1 ]; then
  default_suffix="-linux"
  PK="pk -p"
else
  default_suffix="-baremetal"
  PK=""
fi

path=""
suffix=""

find_binary_in_dir() {
  local search_dir="$1"
  local binary_name="$2"
  local suffix="$3"
  if [ -f "${search_dir}/${binary_name}${suffix}" ]; then
    echo "${search_dir}/"
    return 0
  fi
  for subdir in $(find "${search_dir}" -type d); do
    if [ -f "${subdir}/${binary_name}${suffix}" ]; then
      echo "${subdir}/"
      return 0
    fi
  done
  return 1
}

for dir in optest; do
  base_dir="${ROOT}/sim/output/workloads/${dir}"
  if [ -d "${base_dir}" ]; then
    found_path=$(find_binary_in_dir "${base_dir}" "${binary}" "${default_suffix}")
    if [ $? -eq 0 ]; then
      path="${found_path}"
      suffix="${default_suffix}"
      break
    fi
  fi
done

full_binary_path="${path}${binary}${suffix}"

if [ ! -f "${full_binary_path}" ]; then
  echo "Binary not found: $full_binary_path"
  exit 1
fi

if [ $debug -eq 1 ]; then
  LOG_DIR="${SIM_DIR}/log/${TIMESTAMP}-${binary}-spike-run-log"
  mkdir -p "${LOG_DIR}"
  # spike --extension=${extension} -l --log=${LOG_DIR}/disasm.log \
  if [ -n "$extension" ]; then
    spike --extension=${extension} -l --log=${LOG_DIR}/disasm.log --log-commits \
      $PK "${full_binary_path}" 2>&1 | tee ${LOG_DIR}/stdout.log
  else
    spike -l --log=${LOG_DIR}/disasm.log --log-commits \
      $PK "${full_binary_path}" 2>&1 | tee ${LOG_DIR}/stdout.log
  fi
else
  if [ -n "$extension" ]; then
    echo "spike --extension=${extension} $PK "${full_binary_path}""
    spike --extension=${extension} $PK "${full_binary_path}"
  else
    echo "spike $PK "${full_binary_path}""
    spike $PK "${full_binary_path}"
  fi
fi

