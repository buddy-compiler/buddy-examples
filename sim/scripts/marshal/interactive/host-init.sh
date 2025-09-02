#!/bin/bash

# This script will run on the host from the workload directory
# (e.g. workloads/example-fed) every time the workload is built.
# It is recommended to call into something like a makefile because
# this script may be called multiple times.

ROOT=$(git rev-parse --show-toplevel)

cd $ROOT && source env.sh

echo "Building marshal workload"
cd $ROOT/models/build
make 

cd $ROOT/models/output
rm -rf ./marshal/overlay/root/
mkdir -p ./marshal/overlay/root/
cp -r ./models/* ./marshal/overlay/root/
