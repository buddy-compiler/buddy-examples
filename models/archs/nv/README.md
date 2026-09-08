# NV arch (TOSA → NVIDIA GPU)

Host path for e2e models, adapted from `buddy-mlir/examples/BuddyLeNet` GPU pipeline:

```
tosa → linalg → bufferize → parallel-loops → gpu → nvvm (sm_*) → host .o
```

## Models

Same set as `archs/buckyball`: LeNet, ResNet18, MobileNetV3, YOLO26, Bert, llama2, DeepSeekR1, Qwen3, Gemma4, StableDiffusion, BuddyNext, Whisper, plus MiniMaxH3FL2VA / MiniMaxH3Ref2VA (NV-only).

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

## MiniMax-H3

```bash
export MINIMAX_H3_FL2VA_MODEL_PATH=/path/to/MiniMax-H3/FL2VA   # or repo root
export MINIMAX_H3_REF2VA_MODEL_PATH=/path/to/MiniMax-H3/Ref2VA
export MINIMAX_API_BASE=https://api.minimax.io   # or CN endpoint
export MINIMAX_API_KEY=...
cmake -G Ninja .. -DARCH=nv -DMODEL=minimax_h3_fl2va,minimax_h3_ref2va \
  -DPython3_EXECUTABLE=$PWD/../../../../../../result/bin/python3
ninja buddy-nv-minimax-h3-fl2va-run buddy-nv-minimax-h3-ref2va-run
```

FL2VA: `./buddy-nv-minimax-h3-fl2va-run --mode t2va --prompt "..."`  
Ref2VA: `./buddy-nv-minimax-h3-ref2va-run --prompt "..." --ref-image a.png`  
Both call Context-IR + local Base 768p + Regenerate-2K; missing env/API/weights exits non-zero.
