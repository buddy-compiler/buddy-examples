#!/bin/bash

set -e

ROOT=$(git rev-parse --show-toplevel)
FIRESIM_CONFIG_DIR=$ROOT/sims/firesim/yaml
FIRESIM_DEPLOY_DIR=$ROOT/thirdparty/chipyard/sims/firesim/deploy

firesim infrasetup \
  -a $FIRESIM_CONFIG_DIR/config_hwdb.yaml \
  -b $FIRESIM_CONFIG_DIR/config_build.yaml \
  -r $FIRESIM_CONFIG_DIR/config_build_recipes.yaml \
  -c $FIRESIM_CONFIG_DIR/config_runtime.yaml

firesim runworkload \
  -a $FIRESIM_CONFIG_DIR/config_hwdb.yaml \
  -b $FIRESIM_CONFIG_DIR/config_build.yaml \
  -r $FIRESIM_CONFIG_DIR/config_build_recipes.yaml \
  -c $FIRESIM_CONFIG_DIR/config_runtime.yaml
