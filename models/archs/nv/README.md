# NV arch (TOSA → NVIDIA GPU)

Host path for e2e models, adapted from `buddy-mlir/examples/BuddyLeNet` GPU pipeline:

```
tosa → linalg → bufferize → parallel-loops → gpu → nvvm (sm_*) → host .o
```

## Models

Same set as `archs/buckyball`: LeNet, ResNet18, MobileNetV3, YOLO26, Bert, llama2, DeepSeekR1, Qwen3, Gemma4, StableDiffusion, BuddyNext, Whisper.

Each model has its own `CMakeLists.txt` (LeNet GPU template, adapted). Targets look like `buddy-nv-<model>-run` (BuddyNext: per-kernel `buddy-nv-buddynext-*-run`).

## Prerequisites

Rebuild LLVM/MLIR with CUDA runner (`NVPTX` + `MLIR_ENABLE_CUDA_RUNNER=ON`), then buddy-mlir. Vision models also need `BuddyLibDIP` / Whisper needs `BuddyLibDAP` in the buddy-mlir build.

## Build

```bash
cd bb-tests/workloads/src/ModelTest/e2e/models
mkdir -p build && cd build
cmake -G Ninja .. -DMODEL=lenet,resnet18 -DARCH=nv
ninja buddy-nv-lenet-run
```

Optional: `NV_CUBIN_CHIP=sm_86` (default `sm_80`).

Runtime deps (`images/`, `arg0.data`, …) sync next to the binary. Run from that directory.

On non-NixOS with nix-built `libmlir_cuda_runtime`, you may need:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1 ./buddy-nv-lenet-run
```

(path varies; see host NVIDIA driver).
