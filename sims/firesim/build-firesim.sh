#!/bin/bash

set -e

ROOT=$(git rev-parse --show-toplevel)
FIRESIM_DIR=$ROOT/sims/firesim

source $ROOT/env.sh
firesim buildbitstream \
-a $FIRESIM_DIR/yaml/config_hwdb.yaml \
-b $FIRESIM_DIR/yaml/config_build.yaml \
-r $FIRESIM_DIR/yaml/config_build_recipes.yaml \
-c $FIRESIM_DIR/yaml/config_runtime.yaml
