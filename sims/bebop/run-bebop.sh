#!/bin/bash

ROOT=$(git rev-parse --show-toplevel)

if [ -z "$1" ]; then
  echo "Error: workload name is required"
  echo "Usage: $0 <workload-name>"
  echo "Valid workload-names: lenet-gemmini, resnet-gemmini, mobilenetv3-gemmini, \
     bert-gemmini, stablediffusion-gemmini, llama2-gemmini, deepseekr1-gemmini"
  exit 1
fi
WORKLOAD=$1

if [ $WORKLOAD == "lenet-gemmini" ]; then
  TEST_BINARY_PATH=${ROOT}/models/bin/buddy-gemmini-lenet-run
elif [ $WORKLOAD == "resnet-gemmini" ]; then
  TEST_BINARY_PATH=${ROOT}/models/bin/buddy-gemmini-resnet-run
elif [ $WORKLOAD == "mobilenetv3-gemmini" ]; then
  TEST_BINARY_PATH=${ROOT}/models/bin/buddy-gemmini-mobilenetv3-run
elif [ $WORKLOAD == "bert-gemmini" ]; then
  TEST_BINARY_PATH=${ROOT}/models/bin/buddy-gemmini-bert-run
elif [ $WORKLOAD == "stablediffusion-gemmini" ]; then
  TEST_BINARY_PATH=${ROOT}/models/bin/buddy-gemmini-stablediffusion-run
elif [ $WORKLOAD == "llama2-gemmini" ]; then
  TEST_BINARY_PATH=${ROOT}/models/bin/buddy-gemmini-llama2-run
elif [ $WORKLOAD == "deepseekr1-gemmini" ]; then
  TEST_BINARY_PATH=${ROOT}/models/bin/buddy-gemmini-deepseekr1-run
fi

cd ${ROOT}/thirdparty/bebop/bebop
cargo build --release --bin bebop 

cd ${ROOT}/models/bin
${ROOT}/thirdparty/bebop/bebop/target/release/bebop \
  --step \
  --host gem5 \
  --gem5-mode se \
  --arch gemmini \
  --se-binary ${TEST_BINARY_PATH}

# hello world example
# cargo run --release --bin bebop -- --host gem5 --se-binary ${ROOT}/thirdparty/bebop/host/gem5/test/hello

# cargo run --release --bin bebop -- \
#   --host gem5 \
#   --gem5-mode fs \
#   --fs-kernel "${ROOT}/thirdparty/chipyard/software/firemarshal/images/firechip/interactive/interactive-bin" \
#   --fs-image "${ROOT}/thirdparty/chipyard/software/firemarshal/images/firechip/interactive/interactive.img"