#!/bin/bash

set -e

ROOT=$(git rev-parse --show-toplevel)
MARSHAL_DIR=$ROOT/sims/marshal

source $ROOT/env.sh
cd $MARSHAL_DIR
./marshal -v build interactive.json  && ./marshal -v install interactive.json