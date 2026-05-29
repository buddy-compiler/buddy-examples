# LeNet Directory Guide

This directory contains the source files used to define, train, import, run, and trace the LeNet example model.

This README intentionally documents only checked-in source files and sample inputs. It does not describe generated artifacts such as `.mlir`, `.data`, model checkpoints, or output traces.

## File Overview

### Build and orchestration

- `CMakeLists.txt`
  Defines the local build pipeline for the LeNet model assets. It wires together training, model import, and trace-MLIR generation steps that are consumed by higher-level architecture-specific builds.

### Model definition and PyTorch utilities

- `model.py`
  Defines the PyTorch `LeNet` network architecture used by training, import, inference, and tracing scripts.

- `pytorch-lenet-train.py`
  Trains LeNet on MNIST and saves a model checkpoint. It uses `ToTensor()` plus `Normalize((0.5,), (0.5,))` for input preprocessing.

- `pytorch-lenet-inference.py`
  A simple standalone PyTorch inference example for a sample image. It is useful for quick manual checks of preprocessing and prediction behavior.

- `pytorch-lenet-trace.py`
  Runs the PyTorch model step by step and writes per-layer tensor dumps in NDJSON format. This is the reference trace generator used when comparing runtime behavior against the compiled implementation.

### Import and tracing helpers

- `buddy-lenet-import.py`
  Imports the trained PyTorch model into the Buddy compiler frontend and emits the lowered model representation plus packed parameters for downstream compilation.

- `generate_lenet_trace_mlir.py`
  Inserts trace hooks into the LeNet subgraph so intermediate tensors can be captured during execution.

### Native runtime entry points

- `buddy-lenet-main.cpp`
  Native C++ entry point for normal LeNet execution. It loads the input image, applies the LeNet input normalization expected by the PyTorch model, loads parameters, invokes the compiled forward function, and prints the final classification result.

- `buddy-lenet-trace-main.cpp`
  Native C++ entry point for traced execution. It follows the same execution flow as `buddy-lenet-main.cpp`, but also records the normalized input tensor and relies on injected trace hooks to dump intermediate tensors.

- `lenet-host-trace-runner.cpp`
  A host-side runner for executing the traced subgraph directly from binary tensor files. This is useful for debugging and replaying traced inputs outside the full image-loading path.

### Shared support files

- `include/testutils.h`
  Small utility header shared by the C++ runners. Right now it provides cycle counting support used for simple performance logging.

### Sample inputs

- `images/8.bmp`
  The sample image used by the C++ LeNet runners in this directory.

- `images/3.png`
  A sample image used by the standalone PyTorch inference script.

