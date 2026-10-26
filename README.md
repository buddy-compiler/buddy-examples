# buddy-codesign

Buddy Codesign is a hardware–software co-design platform for AI workloads, built around the Buddy compiler ecosystem. It brings together model preparation, compiler transformations, runtime execution, and hardware evaluation to explore how AI models map onto different computing architectures. The goal is to connect model requirements with hardware capabilities throughout the development process. Quantization, data layout, computation partitioning, memory management, and execution scheduling are developed together with the target hardware, using execution results and traces to guide further refinement.

## Quick Start

**1. Environment Dependencies**

We use Nix Flake as our main build system. If you have not installed nix, install it following the [guide](https://nix.dev/manual/nix/2.28/installation/installing-binary.html), and enable flake following the [wiki](https://nixos.wiki/wiki/Flakes#Enable_flakes). Or you can try the [installer](https://github.com/DeterminateSystems/nix-installer) provided by Determinate Systems, which enables flake by default.

**2. Initialize Environment**
```bash
cd buckyball
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
cd models/archs/nv/mobilenet-v3-small

```

如果没有gpu，x86通路同理

执行该命令xxxxx,顺利的话会看到输出如下

```
cd models/archs/x86/mobilenet-v3-small

```

### step 3. 使用buddy-mlir lower到特定的DSA后端（buckyball/Gemmini）

目前平台支持的DSA：
（表格）

我们先使用buckyball在bemu模拟器上演示pebble设计的



然后我们使用firesim在FPGA上演示buckyball与gemmini


## 如何软硬件协同优化一个模型？

### Lesson 1: Layout 优化


### Lesson 2：量化框架与精度评估


### Lesson 3：子图分割（分配到每个核心）


### Lesson 4：Trace框架


### Lesson 5：子图优化（每个核心内部分割与优化）
