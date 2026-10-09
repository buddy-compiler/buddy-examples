import argparse
import json
from pathlib import Path
import sys
import tomllib
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
    weights = Weights(args.checkpoint)
    torch.manual_seed(42)
    hidden = torch.randn(1, weights.config["hidden_size"])
    settings = tomllib.loads(
        (Path(__file__).parents[1] / "configs/compiler-param.toml").read_text()
    )
    batch = torch.randn(settings["prefill_length"], weights.config["hidden_size"])
    batch[-1].zero_()
    metadata = {}
    for name, stage, target in (
        ("router", weights.router(0), "attention"),
        ("expert", weights.expert(0, 0, 0, 4), "ffn"),
        ("expert_prefill", weights.expert(0, 0, 0, 4), "ffn"),
    ):
        value = batch if name == "expert_prefill" else hidden
        metadata[name] = emit(
            name,
            stage,
            {"hidden": value},
            target,
            {"gate", "up", "down"} if name.startswith("expert") else set(),
            args.output,
            args.compiler_build,
        )
        output = args.output / name
        if name.startswith("expert"):
            with torch.no_grad(), patch("torch.nn.functional.linear", linear):
                stage(value).numpy().tofile(output / "expected.f32")
        else:
            with torch.no_grad():
                normalized, indices, scores = stage(hidden)
            normalized.numpy().tofile(output / "expected.f32")
            indices.numpy().tofile(output / "indices.i64")
            scores.numpy().tofile(output / "scores.f32")
    (args.output / "moe.json").write_text(json.dumps(metadata, indent=2) + "\n")
    hidden.numpy().tofile(args.output / "input.f32")
    batch.numpy().tofile(args.output / "batch-input.f32")
    (args.output / "moe-parameters.h").write_text(
        "#pragma once\n#include <cstddef>\n"
        f"constexpr size_t hiddenSize = {weights.config['hidden_size']};\n"
        f"constexpr size_t topK = {weights.config['num_experts_per_tok']};\n"
        f"constexpr size_t batchSize = {settings['prefill_length']};\n"
        f"constexpr size_t routerFloats = {metadata['router']['floats']};\n"
        f"constexpr size_t expertBytes = {metadata['expert']['bytes']};\n"
    )


if __name__ == "__main__":
    main()
