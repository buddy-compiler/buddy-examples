# GPU (Nvidia) Backend

The NVIDIA backend compiles model computation into GPU kernels through buddy-mlir. The Nix environment provides the compiler dependencies, CUDA toolkit, and host compiler.


## Examples

### buddy-nv-lenet

```bash
cd models
mkdir -p build/nv && cd build/nv
cmake ../../ -G Ninja \
    -DARCH=nv
ninja buddy-nv-lenet-run
```
