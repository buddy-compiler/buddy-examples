import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

from .moe import Weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--compiler-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.compiler_build / "python_packages"))
    from .graph import emit
    from ..quant.reference import linear

    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(7)
    weights = Weights(args.checkpoint)
    stage = weights.attention(0, 0, 4)
    keys = torch.zeros(1, stage.kv_heads, 8, stage.head_dim)
    values = torch.zeros_like(keys)
    metadata = {}
    for name, start, count in (("prefill", 0, 3), ("decode", 3, 1)):
        slots = torch.arange(start, start + count)
        inputs = {
            "hidden": torch.randn(1, count, weights.config["hidden_size"]),
            "keys": keys,
            "values": values,
            "cache_positions": slots,
            "positions": torch.stack((slots, slots * 2 + 7, slots * 3 + 15)),
        }
        metadata[name] = emit(
            name,
            stage,
            inputs,
            "attention",
            {"weights.q", "weights.k", "weights.v", "weights.o"},
            args.output,
            args.compiler_build,
        )
        for key, tensor in inputs.items():
            tensor.numpy().tofile(args.output / name / f"{key}.bin")
        with torch.no_grad(), patch("torch.nn.functional.linear", linear):
            hidden, keys, values = stage(**inputs)
        for key, tensor in (("hidden", hidden), ("keys", keys), ("values", values)):
            tensor.numpy().tofile(args.output / name / f"expected-{key}.bin")
    (args.output / "attention.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "attention-parameters.h").write_text(
        "#pragma once\n#include <cstddef>\n"
        f"constexpr size_t hiddenSize = {weights.config['hidden_size']};\n"
        f"constexpr size_t kvHeads = {stage.kv_heads}, headSize = {stage.head_dim}, capacity = 8;\n"
        f"constexpr size_t prefillFloats = {metadata['prefill']['floats']}, "
        f"prefillBytes = {metadata['prefill']['bytes']};\n"
        f"constexpr size_t decodeFloats = {metadata['decode']['floats']}, "
        f"decodeBytes = {metadata['decode']['bytes']};\n"
    )


if __name__ == "__main__":
    main()
