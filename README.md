# Buckyball software stack

This directory contains the compiler, model recipes, runtime, and serving tools used by Buckyball. The compiler builds on [Buddy MLIR](https://github.com/buddy-compiler/buddy-mlir).

Run the following commands from the Buckyball repository root. Replace `<chip>` and `<model>` with the chip and model recipe you want to use.

```bash
nix develop -c bbdev compiler --build '--chip <chip>'
nix develop -c bbdev model --build '--chip <chip> --model <model>'
nix develop -c bbdev bebop-bemu --sim '--chip <chip> --model <model>'
```

Use `nix develop -c bbdev --help` to list the available workflows.
