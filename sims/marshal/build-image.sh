#!/bin/bash

set -e

# Check if workload name is provided
if [ -z "$1" ]; then
  echo "Error: workload name is required"
  echo "Usage: $0 <workload-name>"
  echo "Valid workload-names: lenet-gemmini, resnet-gemmini, mobilenetv3-gemmini, \
       bert-gemmini, stable-diffusion-gemmini, llama2-gemmini, deepseekr1-gemmini, \
       qwen3-gemmini, yolo26-gemmini, buddynext-gemmini, cnn-gemmini"
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
    -DMODEL="qwen3" \
    -DARCH="gemmini"
  ninja buddy-gemmini-qwen3-run
elif [ $WORKLOAD == "yolo26-gemmini" ]; then
  cd $ROOT/models
  mkdir -p build && cd build
  cmake -G Ninja .. \
    -DMODEL="yolo26" \
    -DARCH="gemmini"
  ninja buddy-gemmini-yolo26-run
elif [ $WORKLOAD == "buddynext-gemmini" ]; then
  cd $ROOT/models
  mkdir -p build && cd build
  cmake -G Ninja .. \
    -DMODEL="buddynext" \
    -DARCH="gemmini"
  ninja buddy-gemmini-buddynext-all-run
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
       qwen3-gemmini, yolo26-gemmini, buddynext-gemmini, cnn-gemmini"
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
  cp $ROOT/models/models/ResNet18/Labels.txt ./
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
elif [ $WORKLOAD == "stable-diffusion-gemmini" ]; then
  mkdir -p $ROOT/models/bin && cd $ROOT/models/bin
  rm -r $ROOT/models/bin/* 2>/dev/null || true
  if [ ! -f $ROOT/models/build/archs/gemmini/StableDiffusion/buddy-gemmini-stable-diffusion-run ]; then
    echo "Error: buddy-gemmini-stable-diffusion-run not found"
    exit 1
  fi
  cp $ROOT/models/build/archs/gemmini/StableDiffusion/buddy-gemmini-stable-diffusion-run ./
  cp $ROOT/models/models/StableDiffusion/arg0_text_encoder.data ./
  cp $ROOT/models/models/StableDiffusion/arg1_text_encoder.data ./
  cp $ROOT/models/models/StableDiffusion/arg0_unet.data ./
  cp $ROOT/models/models/StableDiffusion/arg0_vae.data ./
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
  if [ ! -f $ROOT/models/build/archs/gemmini/DeepSeekR1/buddy-gemmini-deepseekr1-run ]; then
    echo "Error: buddy-gemmini-deepseekr1-run not found"
    exit 1
  fi
  cp $ROOT/models/build/archs/gemmini/DeepSeekR1/buddy-gemmini-deepseekr1-run ./
  cp $ROOT/models/models/DeepSeekR1/arg0.data ./
  cp $ROOT/models/models/DeepSeekR1/vocab.txt ./
elif [ $WORKLOAD == "qwen3-gemmini" ]; then
  mkdir -p $ROOT/models/bin && cd $ROOT/models/bin
  rm -r $ROOT/models/bin/* 2>/dev/null || true
  if [ ! -f $ROOT/models/build/archs/gemmini/Qwen3/buddy-gemmini-qwen3-run ]; then
    echo "Error: buddy-gemmini-qwen3-run not found"
    exit 1
  fi
  cp $ROOT/models/build/archs/gemmini/Qwen3/buddy-gemmini-qwen3-run ./
  cp $ROOT/models/models/Qwen3/arg0_0_6b.data ./
  cp $ROOT/models/models/Qwen3/vocab.txt ./
elif [ $WORKLOAD == "yolo26-gemmini" ]; then
  mkdir -p $ROOT/models/bin && cd $ROOT/models/bin
  rm -r $ROOT/models/bin/* 2>/dev/null || true
  if [ ! -f $ROOT/models/build/archs/gemmini/YOLO26/buddy-gemmini-yolo26-run ]; then
    echo "Error: buddy-gemmini-yolo26-run not found"
    exit 1
  fi
  cp $ROOT/models/build/archs/gemmini/YOLO26/buddy-gemmini-yolo26-run ./
  cp $ROOT/models/models/YOLO26/arg0.data ./
  cp $ROOT/models/models/YOLO26/labels.txt ./
  cp -r $ROOT/models/models/YOLO26/images ./
elif [ $WORKLOAD == "buddynext-gemmini" ]; then
  rm -r $ROOT/models/bin/* 2>/dev/null || true
  BUDDYNEXT_BUILD=$ROOT/models/build/archs/gemmini/BuddyNext
  for kernel in next-embedding next-mhsa-qkv next-mhsa-core next-mhsa-context next-output; do
    binary="buddy-gemmini-buddynext-prefill-${kernel}-run"
    if [ ! -f $BUDDYNEXT_BUILD/$binary ]; then
      echo "Error: $binary not found"
      exit 1
    fi
    mkdir -p $ROOT/models/bin/prefill/$kernel
    cp $BUDDYNEXT_BUILD/$binary $ROOT/models/bin/prefill/$kernel/
  done
  for kernel in next-ffn next-norm next-rope next-gqa-attention next-gqa-attention-fusion next-linalg-matmul next-tosa-matmul; do
    binary="buddy-gemmini-buddynext-decode-${kernel}-run"
    if [ ! -f $BUDDYNEXT_BUILD/$binary ]; then
      echo "Error: $binary not found"
      exit 1
    fi
    mkdir -p $ROOT/models/bin/decode/$kernel
    cp $BUDDYNEXT_BUILD/$binary $ROOT/models/bin/decode/$kernel/
  done
elif [ $WORKLOAD == "cnn-gemmini" ]; then
  rm -r $ROOT/models/bin/* 2>/dev/null || true
  mkdir -p $ROOT/models/bin/lenet && cd $ROOT/models/bin/lenet
  if [ ! -f $ROOT/models/build/archs/gemmini/LeNet/buddy-gemmini-lenet-run ]; then
    echo "Error: buddy-gemmini-lenet-run not found"
    exit 1
  fi
  cp $ROOT/models/build/archs/gemmini/LeNet/buddy-gemmini-lenet-run ./
  cp $ROOT/models/models/LeNet/arg0.data ./
  cp -r $ROOT/models/models/LeNet/images ./

  mkdir -p $ROOT/models/bin/resnet18 && cd $ROOT/models/bin/resnet18
  if [ ! -f $ROOT/models/build/archs/gemmini/ResNet18/buddy-gemmini-resnet-run ]; then
    echo "Error: buddy-gemmini-resnet-run not found"
    exit 1
  fi
  cp $ROOT/models/build/archs/gemmini/ResNet18/buddy-gemmini-resnet-run ./
  cp $ROOT/models/models/ResNet18/arg0.data ./
  cp -r $ROOT/models/models/ResNet18/images ./
  cp $ROOT/models/models/ResNet18/Labels.txt ./
  
  mkdir -p $ROOT/models/bin/mobilenetv3 && cd $ROOT/models/bin/mobilenetv3
  if [ ! -f $ROOT/models/build/archs/gemmini/MobileNetV3/buddy-gemmini-mobilenetv3-run ]; then
    echo "Error: buddy-gemmini-mobilenetv3-run not found"
    exit 1
  fi
  cp $ROOT/models/build/archs/gemmini/MobileNetV3/buddy-gemmini-mobilenetv3-run ./
  cp $ROOT/models/models/MobileNetV3/arg0.data ./
  cp -r $ROOT/models/models/MobileNetV3/images ./
  cp $ROOT/models/models/MobileNetV3/Labels.txt ./
else
  echo "Invalid workload name: $WORKLOAD"
  echo "Valid workload names: lenet-gemmini, resnet-gemmini, mobilenetv3-gemmini, \
       bert-gemmini, stable-diffusion-gemmini, llama2-gemmini, deepseekr1-gemmini, \
       qwen3-gemmini, yolo26-gemmini, buddynext-gemmini, cnn-gemmini"
  exit 1
fi

# step 3: build the image
cd $MARSHAL_DIR
./marshal -v build interactive.json  && ./marshal -v install interactive.json
