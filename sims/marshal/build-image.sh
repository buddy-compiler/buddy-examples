#!/bin/bash

set -e

# Check if workload name is provided
if [ -z "$1" ]; then
  echo "Error: workload name is required"
  echo "Usage: $0 <workload-name>"
  echo "Valid workload-names: lenet-gemmini, resnet-gemmini, mobilenetv3-gemmini, \
       bert-gemmini, stable-diffusion-gemmini, llama2-gemmini, deepseekr1-gemmini, \
       qwen3-gemmini, yolo26-gemmini, cnn-gemmini"
  exit 1
fi

WORKLOAD=$1

ROOT=$(git rev-parse --show-toplevel)
MARSHAL_DIR=$ROOT/sims/marshal

source $ROOT/env.sh

# Preload conda libstdc++ for MLIR Python (GLIBCXX_3.4.29). 
# export LD_PRELOAD=$(conda info --base)/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}

# step 1: build workload 
if [ $WORKLOAD == "lenet-gemmini" ]; then
  cd $ROOT/models
  mkdir -p build && cd build
  cmake -G Ninja .. \
    -DMODEL="lenet" \
    -DARCH="gemmini"
  ninja buddy-gemmini-lenet-run
elif [ $WORKLOAD == "resnet-gemmini" ]; then
  cd $ROOT/models
  mkdir -p build && cd build
  cmake -G Ninja .. \
    -DMODEL="resnet18" \
    -DARCH="gemmini"
  ninja buddy-gemmini-resnet-run
elif [ $WORKLOAD == "mobilenetv3-gemmini" ]; then
  cd $ROOT/models
  mkdir -p build && cd build
  cmake -G Ninja .. \
    -DMODEL="mobilenetv3" \
    -DARCH="gemmini"
  ninja buddy-gemmini-mobilenetv3-run
elif [ $WORKLOAD == "bert-gemmini" ]; then
  cd $ROOT/models
  mkdir -p build && cd build
  cmake -G Ninja .. \
    -DMODEL="bert" \
    -DARCH="gemmini"
  ninja buddy-gemmini-bert-run
elif [ $WORKLOAD == "stable-diffusion-gemmini" ]; then
  cd $ROOT/models
  mkdir -p build && cd build
  cmake -G Ninja .. \
    -DMODEL="stable-diffusion" \
    -DARCH="gemmini"
  ninja buddy-gemmini-stable-diffusion-run
elif [ $WORKLOAD == "llama2-gemmini" ]; then
  cd $ROOT/models
  mkdir -p build && cd build
  cmake -G Ninja .. \
    -DMODEL="llama2" \
    -DARCH="gemmini"
  ninja buddy-gemmini-llama2-run
elif [ $WORKLOAD == "deepseekr1-gemmini" ]; then
  cd $ROOT/models
  mkdir -p build && cd build
  cmake -G Ninja .. \
    -DMODEL="deepseekr1" \
    -DARCH="gemmini"
  ninja buddy-gemmini-deepseekr1-run
elif [ $WORKLOAD == "qwen3-gemmini" ]; then
  cd $ROOT/models
  mkdir -p build && cd build
  cmake -G Ninja .. \
    -DMODEL="qwen3-8b" \
    -DARCH="gemmini"
  ninja buddy-gemmini-qwen3-run
elif [ $WORKLOAD == "yolo26-gemmini" ]; then
  cd $ROOT/models
  mkdir -p build && cd build
  cmake -G Ninja .. \
    -DMODEL="yolo26" \
    -DARCH="gemmini"
  ninja buddy-gemmini-yolo26-run
elif [ $WORKLOAD == "cnn-gemmini" ]; then
  cd $ROOT/models
  mkdir -p build && cd build
  cmake -G Ninja .. \
    -DMODEL="lenet,resnet18,mobilenetv3" \
    -DARCH="gemmini"
  ninja buddy-gemmini-lenet-run buddy-gemmini-resnet-run buddy-gemmini-mobilenetv3-run
else
  echo "Invalid workload name: $WORKLOAD"
  echo "Valid workload names: lenet-gemmini, resnet-gemmini, mobilenetv3-gemmini, \
       bert-gemmini, stable-diffusion-gemmini, llama2-gemmini, deepseekr1-gemmini, \
       qwen3-gemmini, yolo26-gemmini, cnn-gemmini"
  exit 1
fi

# step 2: copy the binary and necessary files to the image
if [ $WORKLOAD == "lenet-gemmini" ]; then
  mkdir -p $ROOT/models/bin && cd $ROOT/models/bin
  rm -r $ROOT/models/bin/* 2>/dev/null || true
  if [ ! -f $ROOT/models/build/archs/gemmini/lenet/buddy-gemmini-lenet-run ]; then
    echo "Error: buddy-gemmini-lenet-run not found"
    exit 1
  fi
  cp $ROOT/models/build/archs/gemmini/lenet/buddy-gemmini-lenet-run ./
  cp $ROOT/models/models/lenet/arg0.data ./
  cp -r $ROOT/models/models/lenet/images ./
elif [ $WORKLOAD == "resnet-gemmini" ]; then
  mkdir -p $ROOT/models/bin && cd $ROOT/models/bin
  rm -r $ROOT/models/bin/* 2>/dev/null || true
  if [ ! -f $ROOT/models/build/archs/gemmini/resnet18/buddy-gemmini-resnet-run ]; then
    echo "Error: buddy-gemmini-resnet-run not found"
    exit 1
  fi
  cp $ROOT/models/build/archs/gemmini/resnet18/buddy-gemmini-resnet-run ./
  cp $ROOT/models/models/resnet18/arg0.data ./
  cp -r $ROOT/models/models/resnet18/images ./
  cp $ROOT/models/models/resnet18/Labels.txt ./
elif [ $WORKLOAD == "mobilenetv3-gemmini" ]; then
  mkdir -p $ROOT/models/bin && cd $ROOT/models/bin
  rm -r $ROOT/models/bin/* 2>/dev/null || true
  if [ ! -f $ROOT/models/build/archs/gemmini/mobilenet-v3-small/buddy-gemmini-mobilenetv3-run ]; then
    echo "Error: buddy-gemmini-mobilenetv3-run not found"
    exit 1
  fi
  cp $ROOT/models/build/archs/gemmini/mobilenet-v3-small/buddy-gemmini-mobilenetv3-run ./
  cp $ROOT/models/models/mobilenet-v3-small/arg0.data ./
  cp -r $ROOT/models/models/mobilenet-v3-small/images ./
  cp $ROOT/models/models/mobilenet-v3-small/Labels.txt ./
elif [ $WORKLOAD == "bert-gemmini" ]; then
  mkdir -p $ROOT/models/bin && cd $ROOT/models/bin
  rm -r $ROOT/models/bin/* 2>/dev/null || true
  if [ ! -f $ROOT/models/build/archs/gemmini/bert/buddy-gemmini-bert-run ]; then
    echo "Error: buddy-gemmini-bert-run not found"
    exit 1
  fi
  cp $ROOT/models/build/archs/gemmini/bert/buddy-gemmini-bert-run ./
  cp $ROOT/models/models/bert/arg0.data ./
  cp $ROOT/models/models/bert/arg1.data ./
  cp $ROOT/models/models/bert/vocab.txt ./
elif [ $WORKLOAD == "stable-diffusion-gemmini" ]; then
  mkdir -p $ROOT/models/bin && cd $ROOT/models/bin
  rm -r $ROOT/models/bin/* 2>/dev/null || true
  if [ ! -f $ROOT/models/build/archs/gemmini/stable-diffusion/buddy-gemmini-stable-diffusion-run ]; then
    echo "Error: buddy-gemmini-stable-diffusion-run not found"
    exit 1
  fi
  cp $ROOT/models/build/archs/gemmini/stable-diffusion/buddy-gemmini-stable-diffusion-run ./
  cp $ROOT/models/models/stable-diffusion/arg0_text_encoder.data ./
  cp $ROOT/models/models/stable-diffusion/arg1_text_encoder.data ./
  cp $ROOT/models/models/stable-diffusion/arg0_unet.data ./
  cp $ROOT/models/models/stable-diffusion/arg0_vae.data ./
elif [ $WORKLOAD == "llama2-gemmini" ]; then
  mkdir -p $ROOT/models/bin && cd $ROOT/models/bin
  rm -r $ROOT/models/bin/* 2>/dev/null || true
  if [ ! -f $ROOT/models/build/archs/gemmini/llama2/buddy-gemmini-llama2-run ]; then
    echo "Error: buddy-gemmini-llama2-run not found"
    exit 1
  fi
  cp $ROOT/models/build/archs/gemmini/llama2/buddy-gemmini-llama2-run ./
  cp $ROOT/models/models/llama2/arg0.data ./
  cp $ROOT/models/models/llama2/vocab.txt ./
elif [ $WORKLOAD == "deepseekr1-gemmini" ]; then
  mkdir -p $ROOT/models/bin && cd $ROOT/models/bin
  rm -r $ROOT/models/bin/* 2>/dev/null || true
  if [ ! -f $ROOT/models/build/archs/gemmini/deepseek-r1-distill-qwen-1.5b/buddy-gemmini-deepseekr1-run ]; then
    echo "Error: buddy-gemmini-deepseekr1-run not found"
    exit 1
  fi
  cp $ROOT/models/build/archs/gemmini/deepseek-r1-distill-qwen-1.5b/buddy-gemmini-deepseekr1-run ./
  cp $ROOT/models/models/deepseek-r1-distill-qwen-1.5b/arg0.data ./
  cp $ROOT/models/models/deepseek-r1-distill-qwen-1.5b/vocab.txt ./
elif [ $WORKLOAD == "qwen3-gemmini" ]; then
  mkdir -p $ROOT/models/bin && cd $ROOT/models/bin
  rm -r $ROOT/models/bin/* 2>/dev/null || true
  if [ ! -f $ROOT/models/build/archs/gemmini/qwen3-8b/buddy-gemmini-qwen3-run ]; then
    echo "Error: buddy-gemmini-qwen3-run not found"
    exit 1
  fi
  cp $ROOT/models/build/archs/gemmini/qwen3-8b/buddy-gemmini-qwen3-run ./
  cp $ROOT/models/models/qwen3-8b/arg0_0_6b.data ./
  cp $ROOT/models/models/qwen3-8b/vocab.txt ./
elif [ $WORKLOAD == "yolo26-gemmini" ]; then
  mkdir -p $ROOT/models/bin && cd $ROOT/models/bin
  rm -r $ROOT/models/bin/* 2>/dev/null || true
  if [ ! -f $ROOT/models/build/archs/gemmini/yolo26n/buddy-gemmini-yolo26-run ]; then
    echo "Error: buddy-gemmini-yolo26-run not found"
    exit 1
  fi
  cp $ROOT/models/build/archs/gemmini/yolo26n/buddy-gemmini-yolo26-run ./
  cp $ROOT/models/models/yolo26n/arg0.data ./
  cp $ROOT/models/models/yolo26n/labels.txt ./
  cp -r $ROOT/models/models/yolo26n/images ./
elif [ $WORKLOAD == "cnn-gemmini" ]; then
  rm -r $ROOT/models/bin/* 2>/dev/null || true
  mkdir -p $ROOT/models/bin/lenet && cd $ROOT/models/bin/lenet
  if [ ! -f $ROOT/models/build/archs/gemmini/lenet/buddy-gemmini-lenet-run ]; then
    echo "Error: buddy-gemmini-lenet-run not found"
    exit 1
  fi
  cp $ROOT/models/build/archs/gemmini/lenet/buddy-gemmini-lenet-run ./
  cp $ROOT/models/models/lenet/arg0.data ./
  cp -r $ROOT/models/models/lenet/images ./

  mkdir -p $ROOT/models/bin/resnet18 && cd $ROOT/models/bin/resnet18
  if [ ! -f $ROOT/models/build/archs/gemmini/resnet18/buddy-gemmini-resnet-run ]; then
    echo "Error: buddy-gemmini-resnet-run not found"
    exit 1
  fi
  cp $ROOT/models/build/archs/gemmini/resnet18/buddy-gemmini-resnet-run ./
  cp $ROOT/models/models/resnet18/arg0.data ./
  cp -r $ROOT/models/models/resnet18/images ./
  cp $ROOT/models/models/resnet18/Labels.txt ./
  
  mkdir -p $ROOT/models/bin/mobilenetv3 && cd $ROOT/models/bin/mobilenetv3
  if [ ! -f $ROOT/models/build/archs/gemmini/mobilenet-v3-small/buddy-gemmini-mobilenetv3-run ]; then
    echo "Error: buddy-gemmini-mobilenetv3-run not found"
    exit 1
  fi
  cp $ROOT/models/build/archs/gemmini/mobilenet-v3-small/buddy-gemmini-mobilenetv3-run ./
  cp $ROOT/models/models/mobilenet-v3-small/arg0.data ./
  cp -r $ROOT/models/models/mobilenet-v3-small/images ./
  cp $ROOT/models/models/mobilenet-v3-small/Labels.txt ./
else
  echo "Invalid workload name: $WORKLOAD"
  echo "Valid workload names: lenet-gemmini, resnet-gemmini, mobilenetv3-gemmini, \
       bert-gemmini, stable-diffusion-gemmini, llama2-gemmini, deepseekr1-gemmini, \
       qwen3-gemmini, yolo26-gemmini, cnn-gemmini"
  exit 1
fi

# step 3: build the image
cd $MARSHAL_DIR
./marshal -v build interactive.json  && ./marshal -v install interactive.json
