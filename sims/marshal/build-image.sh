#!/bin/bash

set -e

# Check if workload name is provided
if [ -z "$1" ]; then
  echo "Error: workload name is required"
  echo "Usage: $0 <workload-name>"
  echo "Valid workload-names: lenet-gemmini"
  exit 1
fi

WORKLOAD=$1

ROOT=$(git rev-parse --show-toplevel)
MARSHAL_DIR=$ROOT/sims/marshal

source $ROOT/env.sh

# step 1: build workload 
if [ $WORKLOAD == "lenet-gemmini" ]; then
  cd $ROOT/models
  mkdir -p build && cd build
  cmake -G Ninja .. \
    -DMODEL="lenet" \
    -DARCH="gemmini"
  ninja buddy-gemmini-lenet-run
else
  echo "Invalid workload name: $WORKLOAD"
  echo "Valid workload names: lenet-gemmini"
  exit 1
fi

# step 2: copy the binary and necessary files to the image
if [ $WORKLOAD == "lenet-gemmini" ]; then
  mkdir -p $ROOT/models/bin && cd $ROOT/models/bin
  rm -r $ROOT/models/bin/* 2>/dev/null || true
  if [ ! -f $ROOT/models/build/archs/gemmini/LeNet/buddy-gemmini-lenet-run ]; then
    echo "Error: buddy-gemmini-lenet-run not found"
    exit 1
  fi
  cp $ROOT/models/build/archs/gemmini/LeNet/buddy-gemmini-lenet-run ./
  cp $ROOT/models/models/LeNet/arg0.data ./
  cp -r $ROOT/models/models/LeNet/images ./
else
  echo "Invalid workload name: $WORKLOAD"
  echo "Valid workload names: lenet-gemmini"
  exit 1
fi


# step 3: build the image
cd $MARSHAL_DIR
./marshal -v build interactive.json  && ./marshal -v install interactive.json
