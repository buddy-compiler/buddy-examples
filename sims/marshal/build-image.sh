#!/bin/bash

set -e

# Check if workload name is provided
if [ -z "$1" ]; then
  echo "Error: workload name is required"
  echo "Usage: $0 <workload-name>"
  echo "Valid workload-names: lenet-gemmini, resnet-gemmini, mobilenetv3-gemmini, \
       bert-gemmini, stablediffusion-gemmini, llama2-gemmini, deepseekr1-gemmini"
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
elif [ $WORKLOAD == "stablediffusion-gemmini" ]; then
  cd $ROOT/models
  mkdir -p build && cd build
  cmake -G Ninja .. \
    -DMODEL="stablediffusion" \
    -DARCH="gemmini"
  ninja buddy-gemmini-stablediffusion-run
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
else
  echo "Invalid workload name: $WORKLOAD"
  echo "Valid workload names: lenet-gemmini, resnet-gemmini, mobilenetv3-gemmini, \
       bert-gemmini, stablediffusion-gemmini, llama2-gemmini, deepseekr1-gemmini"
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
elif [ $WORKLOAD == "resnet-gemmini" ]; then
  mkdir -p $ROOT/models/bin && cd $ROOT/models/bin
  rm -r $ROOT/models/bin/* 2>/dev/null || true
  if [ ! -f $ROOT/models/build/archs/gemmini/ResNet18/buddy-gemmini-resnet-run ]; then
    echo "Error: buddy-gemmini-resnet-run not found"
    exit 1
  fi
  cp $ROOT/models/build/archs/gemmini/ResNet18/buddy-gemmini-resnet-run ./
  cp $ROOT/models/models/ResNet18/arg0.data ./
  cp -r $ROOT/models/models/ResNet18/images ./
elif [ $WORKLOAD == "mobilenetv3-gemmini" ]; then
  mkdir -p $ROOT/models/bin && cd $ROOT/models/bin
  rm -r $ROOT/models/bin/* 2>/dev/null || true
  if [ ! -f $ROOT/models/build/archs/gemmini/MobileNetV3/buddy-gemmini-mobilenetv3-run ]; then
    echo "Error: buddy-gemmini-mobilenetv3-run not found"
    exit 1
  fi
  cp $ROOT/models/build/archs/gemmini/MobileNetV3/buddy-gemmini-mobilenetv3-run ./
  cp $ROOT/models/models/MobileNetV3/arg0.data ./
  cp -r $ROOT/models/models/MobileNetV3/images ./
  cp $ROOT/models/models/MobileNetV3/Labels.txt ./
elif [ $WORKLOAD == "bert-gemmini" ]; then
  mkdir -p $ROOT/models/bin && cd $ROOT/models/bin
  rm -r $ROOT/models/bin/* 2>/dev/null || true
  if [ ! -f $ROOT/models/build/archs/gemmini/Bert/buddy-gemmini-bert-run ]; then
    echo "Error: buddy-gemmini-bert-run not found"
    exit 1
  fi
  cp $ROOT/models/build/archs/gemmini/Bert/buddy-gemmini-bert-run ./
  cp $ROOT/models/models/Bert/arg0.data ./
  cp $ROOT/models/models/Bert/arg1.data ./
  cp $ROOT/models/models/Bert/vocab.txt ./
elif [ $WORKLOAD == "stablediffusion-gemmini" ]; then
  mkdir -p $ROOT/models/bin && cd $ROOT/models/bin
  rm -r $ROOT/models/bin/* 2>/dev/null || true
  if [ ! -f $ROOT/models/build/archs/gemmini/StableDiffusion/buddy-gemmini-stablediffusion-run ]; then
    echo "Error: buddy-gemmini-stablediffusion-run not found"
    exit 1
  fi
  cp $ROOT/models/build/archs/gemmini/StableDiffusion/buddy-gemmini-stablediffusion-run ./
  cp $ROOT/models/models/StableDiffusion/arg0_text_encoder.data ./
  cp $ROOT/models/models/StableDiffusion/arg1_text_encoder.data ./
  cp $ROOT/models/models/StableDiffusion/arg0_unet.data ./
  cp $ROOT/models/models/StableDiffusion/arg0_vae.data ./
elif [ $WORKLOAD == "llama2-gemmini" ]; then
  mkdir -p $ROOT/models/bin && cd $ROOT/models/bin
  rm -r $ROOT/models/bin/* 2>/dev/null || true
  if [ ! -f $ROOT/models/build/archs/gemmini/Llama2/buddy-gemmini-llama2-run ]; then
    echo "Error: buddy-gemmini-llama2-run not found"
    exit 1
  fi
  cp $ROOT/models/build/archs/gemmini/Llama2/buddy-gemmini-llama2-run ./
  cp $ROOT/models/models/Llama2/arg0.data ./
  cp $ROOT/models/models/llama2/vocab.txt ./
elif [ $WORKLOAD == "deepseekr1-gemmini" ]; then
  mkdir -p $ROOT/models/bin && cd $ROOT/models/bin
  rm -r $ROOT/models/bin/* 2>/dev/null || true
  if [ ! -f $ROOT/models/build/archs/gemmini/DeepSeekR1/buddy-gemmini-deepseekr1-run ]; then
    echo "Error: buddy-gemmini-deepseekr1-run not found"
    exit 1
  fi
  cp $ROOT/models/build/archs/gemmini/DeepSeekR1/buddy-gemmini-deepseekr1-run ./
  cp $ROOT/models/models/DeepSeekR1/arg0.data ./
  cp $ROOT/models/models/DeepSeekR1/vocab.txt ./
else
  echo "Invalid workload name: $WORKLOAD"
  echo "Valid workload names: lenet-gemmini, resnet-gemmini, mobilenetv3-gemmini, \
       bert-gemmini, stablediffusion-gemmini, llama2-gemmini, deepseekr1-gemmini"
  exit 1
fi


# step 3: build the image
cd $MARSHAL_DIR
./marshal -v build interactive.json  && ./marshal -v install interactive.json
