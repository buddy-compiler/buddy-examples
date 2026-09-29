# ModelTest Common Libraries

This directory provides common libraries shared across different ModelTest architectures.

## Available Libraries

### 1. ModelTestCRunnerUtils
- **Purpose**: Provides MLIR runtime utilities like `memrefCopy`
- **Source**: `${MODELTEST_LIB_DIR}/CRunnerUtils.cpp`
- **Target**: Cross-platform (compiled for target architecture)

### 2. ModelTestDIP_riscv
- **Purpose**: Provides **Digital Image Processing** operations compiled for RISC-V
- **Source**: `${BUDDY_MLIR_DIR}/frontend/Interfaces/lib/DIP.mlir`
- **Target**: RISC-V (riscv64 with +buddyext,+D)
- **Used by**: Vision models (resnet18, mobilenet-v3-small, stable-diffusion, etc.)
- **Functions included**:
  - `_mlir_ciface_resize_4d_nchw_nearest_neighbour_interpolation`
  - `_mlir_ciface_resize_4d_nchw_bilinear_interpolation`
  - `_mlir_ciface_rotate_4d_nchw`
  - And other DIP operations (corr_2d, morphology, etc.)

### 3. ModelTestDAP_riscv
- **Purpose**: Provides **Digital Audio Processing** operations compiled for RISC-V
- **Source**: `${BUDDY_MLIR_DIR}/frontend/Interfaces/lib/DAP.mlir`
- **Target**: RISC-V (riscv64 with +buddyext,+D)
- **Used by**: Audio models (whisper, etc.)
- **Functions included**:
  - FIR filter operations
  - IIR filter operations
  - Biquad filter operations
  - And other DAP operations

## Usage

In your architecture-specific CMakeLists.txt:

### For Vision Models (using DIP)
```cmake
# Link the common libraries
set(YOUR_LIBS YourModelLib ModelTestCRunnerUtils ModelTestDIP_riscv)
target_link_libraries(your-executable ${YOUR_LIBS})
```

### For Audio Models (using DAP)
```cmake
# Link the common libraries
set(YOUR_LIBS YourModelLib ModelTestCRunnerUtils ModelTestDAP_riscv)
target_link_libraries(your-executable ${YOUR_LIBS})
```

## Examples

- **Vision models**: See `archs/gemmini/resnet18/CMakeLists.txt` or `archs/gemmini/mobilenet-v3-small/CMakeLists.txt`
- **Audio models**: See `archs/gemmini/whisper/CMakeLists.txt`

## Notes

- These libraries are automatically built when you add `add_subdirectory(lib)` in the parent CMakeLists.txt
- Both DIP and DAP libraries are specifically compiled for RISC-V with Gemmini extensions
- **DIP vs DAP**:
  - DIP = Digital Image Processing (for vision models: ResNet, MobileNet, stable-diffusion)
  - DAP = Digital Audio Processing (for audio models: whisper)
- If you need these libraries for other architectures, you can add additional custom commands following the same pattern
