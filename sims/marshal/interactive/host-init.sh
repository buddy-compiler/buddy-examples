#!/bin/bash

# This script will run on the host from the workload directory
# (e.g. workloads/example-fed) every time the workload is built.
# It is recommended to call into something like a makefile because
# this script may be called multiple times.

ROOT=$(git rev-parse --show-toplevel)
MARSHAL_DIR=$ROOT/sims/marshal

cd $ROOT && source env.sh

# echo "Building marshal workload"
# cd $ROOT/models/build
# make 

cd $ROOT/models/bin
rm -r $MARSHAL_DIR/overlay/root/* 2>/dev/null || true
mkdir -p $MARSHAL_DIR/overlay/root/
cp -r ./* $MARSHAL_DIR/overlay/root/
