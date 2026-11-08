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

