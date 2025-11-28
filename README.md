# buddy-examples

This repository demonstrates buddy-mlir capabilities through selected representative workloads running on Chipyard 1.13.1 platform. These example models are lowered to Gemmini Dialect by buddy-mlir, where Gemmini is a DNN accelerator integrated in Chipyard. The workloads are simulated using tools provided by Chipyard: Spike functional simulator, and FireSim FPGA-accelerated simulation.

## Quick Start

1. Environment Dependencies

Before getting started, please ensure your system meets the following dependency requirements:

- Anaconda/Miniconda (environment management)
- Ninja Build System

2. Initialize repository in one step:

```bash
git clone https://github.com/buddy-compiler/buddy-examples.git
cd buddy-examples
./scripts/init.sh
```

3. Activate buddy-examples' environment

```bash
cd buddy-examples
source ./env.sh
``` 

4. Build hardware in FireSim

We have tested these examples on VCU118 and Alveo U280 platforms.

- Configuration setup
    - replace [`default_build_dir`](sims/firesim/yaml/config_build.yaml) with your desired directory path for storing generated bitstream files.
    - replace [`default_simulation_dir`](sims/firesim/yaml/config_runtime.yaml) with your desired directory path for storing simulation-related files.

- Generate bitstream and run simulation

```bash
./sims/firesim/build-firesim.sh
```

Build the FPGA bitstream and launch FireSim simulations. The bitstream generation process may take several hours depending on your hardware configuration.

```bash
cd buddy-examples
source ./env.sh
``` 

**Note:** This guide assumes you have basic knowledge of FireSim. For detailed FireSim configuration instructions, please refer to the [official FireSim documentation](https://docs.fires.im).


Then, you can start running the examples below.

## Examples

### LeNet-Gemmini
This example uses the LeNet model with the MNIST dataset. Note that the build process will automatically download the dataset and train the model locally before building workloads, which may take some time.

1. Build Workloads

```bash
cd models
mkdir -p build && cd build
cmake -G Ninja .. \
    -DMODEL="lenet" \
    -DARCH="gemmini"
ninja buddy-gemmini-lenet-run
```

2.

### ResNet18
```bash
mkdir build && cd build
cmake -G Ninja .. \
    -DMODEL="lenet,resnet18,mobilenetv3,bert,stablediffusion,llama2,deepseekr1" \
    -DARCH="gemmini"
ninja
```
