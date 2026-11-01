# buddy-codesign

Buddy Codesign is a hardware–software co-design platform for AI workloads, built around the Buddy compiler ecosystem. It brings together model preparation, compiler transformations, runtime execution, and hardware evaluation to explore how AI models map onto different computing architectures. The goal is to connect model requirements with hardware capabilities throughout the development process. Quantization, data layout, computation partitioning, memory management, and execution scheduling are developed together with the target hardware, using execution results and traces to guide further refinement.

## Quick Start

**1. Environment Dependencies**

We use Nix Flake as our main build system. If you have not installed nix, install it following the [guide](https://nix.dev/manual/nix/2.28/installation/installing-binary.html), and enable flake following the [wiki](https://nixos.wiki/wiki/Flakes#Enable_flakes). Or you can try the [installer](https://github.com/DeterminateSystems/nix-installer) provided by Determinate Systems, which enables flake by default.

**2. Initialize Environment**
```bash
cd buddy-codesign
./scripts/nix/build-all.sh
```

After the first time installation, you can enter the environment anytime by running:

```bash
nix develop
```

Then, you can start running the examples below.

## 如何适配一个新模型？

下面是我们认为一个适合新手入门的路线，里面很多步骤并不是必须执行

### step 1. 导入模型并查看原始计算图

模型定义、权重准备和原始计算图导出统一放在 `models/models` 下。以 `mobilenet-v3-small` 为例，先运行 `buddy-pytorch-mobilenet-v3-small-prepare.py` 下载并保存预训练权重，再运行 `buddy-pytorch-mobilenet-v3-small-graph.py`，通过 Buddy 前端导入模型并导出未插桩的原始计算图（Buddy Graph）。

```
cd buddy-codesign
nix develop

cd models/models/mobilenet-v3-small
cmake ../../ -G Ninja \
    -DARCH=nv

ninja mobilenet-v3-small-prepare
ninja buddy-mobilenet-v3-small-graph
```

执行成功后，模型权重保存在当前目录的 `output/mobilenet-v3-small/model.pth`，计算图输出到 `output/buddy-graph.txt`。可以打开该文件，查看模型的算子及其连接关系。

### step 2. 使用通用设备进行验证（CPU/GPU）

完成模型权重准备后，可以使用 NVIDIA GPU 或 x86 CPU 等通用通路验证模型编译。我们需要将模型编译并打包为 .rax，通过 buddy-cli 加载和执行。

以下命令分别从仓库根目录开始，在 Nix 开发环境中执行。

使用 NVIDIA GPU 时，先配置并构建 NV 后端：

```
cmake -S models -B output/models/nv/mobilenet-v3-small -G Ninja -DARCH=nv -DMODEL=mobilenet-v3-small
cmake --build output/models/nv/mobilenet-v3-small --target buddy-nv-mobilenet-v3-small-rax-origin-build
thirdparty/buddy-mlir/build/bin/buddy-cli --model models/archs/nv/mobilenet-v3-small/origin/output/mobilenet-v3-small.rax --image /absolute/path/to/image.png
```

如果没有gpu，x86通路同理

例如，使用仓库中的 LeNet 检查点和测试图片：

```
cmake -S models -B output/models/x86/lenet -G Ninja -DARCH=x86 -DMODEL=lenet
cmake --build output/models/x86/lenet --target buddy-x86-lenet-rax-origin-build
thirdparty/buddy-mlir/build/bin/buddy-cli --model models/archs/x86/lenet/origin/output/lenet.rax --image models/models/lenet/images/8.bmp
# Classification: 8
```

### step 3. 使用buddy-mlir lower到特定的DSA后端（buckyball/Gemmini）

模拟器入口分别使用带 Gemmini 扩展的 Spike 和 Bebop BEMU。在 Nix 环境中安装：

```bash
./sims/spike/install.sh
./sims/bemu/install.sh
```

执行已经编译的 ELF；模型的 RAX 包需要由 RISC-V Linux 中的 buddy-cli 加载，不能直接作为裸机 ELF：

```bash
output/spike/bin/spike --extlib=output/spike/lib/libgemmini.so --extension=gemmini /absolute/path/to/gemmini-baremetal.elf
```

BEMU 通过 bbdev 运行（在 Buckyball 仓库根目录）：

```bash
nix develop -c bbdev bebop-bemu --sim "--design toy --device ToyVerilatorConfig --binary <workload>"
```

Spike 安装使用 Chipyard 固定版本的模拟器、Gemmini 扩展和 proxy kernel，产物位于 `output/spike`。

我们先使用buckyball在bemu模拟器上演示pebble设计的



FireSim 的依赖安装和 FPGA RTL 生成不需要连接开发板：

```bash
./sims/firesim/install.sh
make -C platforms/thirdparty/chipyard/sims/firesim/sim verilog \
  FIRESIM_ENV_SOURCED=1 TARGET_PROJECT=firesim \
  SBT="$JAVA_HOME/bin/java -Xmx8G -jar platforms/thirdparty/chipyard/scripts/sbt-launch.jar" \
  TARGET_PROJECT_MAKEFRAG=$PWD/sims/firesim/makefrag/firesim PLATFORM_CONFIG_PACKAGE=firesim.configs \
  TARGET_CONFIG=FireSimGemminiVCU118Config PLATFORM=xilinx_vcu118 PLATFORM_CONFIG=BaseXilinxVCU118Config
```

上板时另行提供目标机器、bitstream 和运行配置。


## 如何软硬件协同优化一个模型？

### Lesson 1: Layout 优化


### Lesson 2：量化框架与精度评估


### Lesson 3：子图分割（分配到每个核心）


### Lesson 4：Trace框架


### Lesson 5：子图优化（每个核心内部分割与优化）
