#!/bin/bash

ROOT=$(git rev-parse --show-toplevel)
SIM_DIR=${ROOT}/sim
WAVEFORM=""
WAVEFORM_DIR="${SIM_DIR}/output/waveforms"
mkdir -p "${WAVEFORM_DIR}"

TIMESTAMP=$(date +%Y-%m-%d-%H-%M)

help () {
  echo "Run a RISCV Gemmini program on Verilator, a cycle-accurate simulator"
  echo
  echo "Usage: $0 [--pk] [--config] [--debug] [--vcd2fst] BINARY"
  echo
  echo "Options:"
  echo " pk    Run binaries on the proxy kernel, which enables virtual memory"
  echo "     and a few syscalls. If this option is not set, binaries will be"
  echo "     run in baremetal mode."
  echo
  echo " config   --config/-c your scala Config"
  echo
  echo " debug   Use the debug version of the Verilator simulator, which will"
  echo "     output a waveform to \`$WAVEFORM\`."
  echo
  echo " vcd2fst  Compress the VCD waveform to FST format after simulation."
  echo
  echo " BINARY  The RISCV binary that you want to run. This can either be the"
  echo '     name of a program in `software/gemmini-rocc-tests`, or it can'
  echo "     be the full path to a binary you compiled."
  echo
  echo "Examples:"
  echo "     $0 template"
  echo "     $0 --debug template"
  echo "     $0 --pk mvin_mvout"
  echo "     $0 path/to/binary-baremetal"
  echo
  echo 'Note:  Run this command after running `scripts/build-verilator.sh` or'
  echo '     `scripts/build-verilator.sh --debug`.'
  exit
}

if [ $# -le 0 ]; then
  help
fi

pk=0
debug=0
show_help=0
binary=""
vcd2fst=0

while [ $# -gt 0 ] ; do
  case $1 in
  --pk) pk=1 ;;
  -c|--config)
    if [[ -n $2 && $2 != -* ]]; then
    CONFIG="$2"
    shift
    else
    echo "Error: -c or --config need a parameter"
    help
    fi
    ;;
  --debug) debug=1 ;;
  --vcd2fst) vcd2fst=1 ;;
  -h | --help) show_help=1 ;;
  *) binary=$1
  esac

  shift
done

if [ $show_help -eq 1 ]; then
   help
fi

if [ $pk -eq 1 ]; then
  default_suffix="-pk"
  PK="pk -p"
else
  default_suffix="-baremetal"
  PK=""
fi

WAVEFORM="${WAVEFORM_DIR}/${TIMESTAMP}-waveform.vcd"

if [ $debug -eq 1 ]; then
  DEBUG="-debug"
else
  DEBUG=""
fi

path=""
suffix=""

# Recursive search function
find_binary_in_dir() {
  local search_dir="$1"
  local binary_name="$2"
  local suffix="$3"
  
  # Check current directory
  if [ -f "${search_dir}/${binary_name}${suffix}" ]; then
  echo "${search_dir}/"
  return 0
  fi
	# Recursively check all subdirectories
	for subdir in $(find "${search_dir}" -type d); do
	  if [ -f "${subdir}/${binary_name}${suffix}" ]; then
	    echo "${subdir}/"
	    return 0
	  fi
	done
	return 1
}

# Search for binary files in cpu and npu and their subdirectories
# embench cannot be identified due to different naming format, so the corresponding execution script can only use absolute path
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

LOG_DIR="${SIM_DIR}/log/${TIMESTAMP}-${binary}-verilator-run-log"
mkdir -p "${LOG_DIR}"


cd ${ROOT}/sim/output/verilator/

# Replace the original echo statements
echo "Running Verilator simulation with configuration:"
echo "  Simulator: ./simulator-chipyard.harness-${CONFIG}${DEBUG}"
echo "  Binary: ${full_binary_path}"
echo "  PK mode: $([ $pk -eq 1 ] && echo "enabled" || echo "disabled")"
echo "  Debug mode: $([ $debug -eq 1 ] && echo "enabled" || echo "disabled")"
echo "  Waveform: $([ $debug -eq 1 ] && echo "${WAVEFORM}" || echo "disabled")"
echo "  Log directory: ${LOG_DIR}"
echo ""
echo "Command line:"
echo "./simulator-chipyard.harness-${CONFIG}${DEBUG} $PK +permissive \\"
echo $([ $debug -eq 1 ] && echo "  +vcdfile=${WAVEFORM} \\")
echo $([ $debug -eq 1 ] && echo "  +verbose \\")
echo "  +loadmem=${full_binary_path} \\"
echo "  +loadmem_addr=80000000 \\"
echo "  +permissive-off \\"
echo "  ${full_binary_path}"
echo ""

if [ $debug -eq 1 ]; then
  ./simulator-chipyard.harness-${CONFIG}${DEBUG} $PK +permissive \
  +vcdfile=${WAVEFORM} \
  +verbose \
  +loadmem=${full_binary_path} +loadmem_addr=80000000 \
  +permissive-off \
  ${full_binary_path} \
  &> >(tee ${LOG_DIR}/stdout.log) \
  2> >(spike-dasm > ${LOG_DIR}/disasm.log)
else
  ./simulator-chipyard.harness-${CONFIG}${DEBUG} $PK +permissive \
  +loadmem=${full_binary_path} +loadmem_addr=80000000 \
  +permissive-off \
  ${full_binary_path} \
  &> >(tee ${LOG_DIR}/stdout.log)
fi




# If debug mode is enabled and waveform file conversion is needed
if [ $debug -eq 1 ] && [ $vcd2fst -eq 1 ]; then
  echo "Converting VCD waveform to FST format..."
  FST_WAVEFORM="${WAVEFORM%.vcd}.fst"
  vcd2fst -v "${WAVEFORM}" -f "${FST_WAVEFORM}"
  rm -rf "${WAVEFORM}"
  if [ $? -eq 0 ]; then
  echo "Waveform conversion successful: ${FST_WAVEFORM}"
  else
  echo "Warning: Waveform conversion failed."
  fi
fi
