#!/bin/bash

ROOT=$(git rev-parse --show-toplevel)
source ${ROOT}/env.sh

cd ${ROOT}/thirdparty/buddy-mlir/build || { echo "Cannot enter the directory: ${ROOT}/thirdparty/buddy-mlir/build"; exit 1; }
ninja -j$(nproc)
