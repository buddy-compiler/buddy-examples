import argparse
import json
from pathlib import Path
import sys
import tomllib

import torch

from .moe import Weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-config", type=Path, required=True)
    parser.add_argument("--compiler-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system-config", type=Path, required=True)
    args = parser.parse_args()
    args.checkpoint = Path(json.loads(args.checkpoint_config.read_text())["path"])
    sys.path.insert(0, str(args.compiler_build / "python_packages"))
    from .graph import emit

    design = Path(__file__).resolve().parent.parent
    settings = tomllib.loads((design / "configs/compiler-param.toml").read_text())
    system = tomllib.loads(args.system_config.read_text())
    parts = len(system["thinker"]["tensor_tiles"])
    weights = Weights(args.checkpoint)
    config = weights.config
    hidden = config["hidden_size"]
    cache_shape = (1, config["num_key_value_heads"] // parts, settings["cache_length"], config["head_dim"])
    torch.set_num_threads(4)
    torch.manual_seed(42)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {"stages": {}, "config": config, "system": system, **settings}
    for phase, count in (("prefill", settings["prefill_length"]), ("decode", 1)):
        name = f"{phase}_attention"
        inputs = {"hidden": torch.randn(1, count, hidden), "keys": torch.zeros(cache_shape),
                  "values": torch.zeros(cache_shape), "cache_positions": torch.arange(count),
                  "positions": torch.arange(count).repeat(3, 1)}
        metadata["stages"][name] = emit(name, weights.attention(0, 0, parts), inputs, "attention",
            {"weights.q", "weights.k", "weights.v", "weights.o"}, args.output, args.compiler_build)
        name = f"{phase}_router"
        metadata["stages"][name] = emit(name, weights.router(0), {"hidden": torch.randn(count, hidden)},
            "attention", set(), args.output, args.compiler_build)
        name = f"{phase}_embedding"
        metadata["stages"][name] = emit(name, weights.embedding(0, parts),
            {"tokens": torch.zeros(1, count, dtype=torch.int64), "begin": torch.zeros(1, dtype=torch.int64)},
            "attention", set(), args.output, args.compiler_build)
        name = f"{phase}_norm"
        metadata["stages"][name] = emit(name, weights.norm(), {"hidden": torch.randn(count, hidden)},
            "attention", set(), args.output, args.compiler_build)
    metadata["stages"]["expert"] = emit("expert", weights.expert(0, 0, 0, parts),
        {"hidden": torch.randn(1, hidden)}, "ffn", {"gate", "up", "down"}, args.output, args.compiler_build)
    metadata["stages"]["output"] = emit("output", weights.output(0, parts),
        {"hidden": torch.randn(1, hidden)}, "ffn", {"weight"}, args.output, args.compiler_build)
    manifest = args.output / "kernels.json"
    contents = json.dumps(metadata, indent=2) + "\n"
    if not manifest.exists() or manifest.read_text() != contents:
        manifest.write_text(contents)
    constants = {"hiddenSize": hidden, "layers": config["num_hidden_layers"],
                 "expertsCount": config["num_experts"], "topK": config["num_experts_per_tok"],
                 "parts": parts, "kvHeads": cache_shape[1], "headSize": config["head_dim"],
                 "vocabulary": config["vocab_size"], "vocabularyPart": config["vocab_size"] // parts,
                 "prefillLength": settings["prefill_length"], "cacheLength": settings["cache_length"],
                 "workspaceBytes": 64 * 1024 * 1024}
    (args.output / "model-parameters.h").write_text("".join(
        f"constexpr size_t {name} = {value};\n" for name, value in constants.items()))


if __name__ == "__main__":
    main()
