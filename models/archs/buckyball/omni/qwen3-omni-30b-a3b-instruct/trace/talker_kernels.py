import argparse
import json
from pathlib import Path
import sys
import tomllib

import torch

from .talker import Weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-config", type=Path, required=True)
    parser.add_argument("--compiler-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.compiler_build / "python_packages"))
    from .graph import emit

    source = Weights(Path(json.loads(args.checkpoint_config.read_text())["path"]))
    settings = tomllib.loads(
        (Path(__file__).parents[1] / "configs/compiler-param.toml").read_text()
    )
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    cache_buckets = []
    capacity = 16
    while capacity < settings["cache_length"]:
        cache_buckets.append(capacity)
        capacity *= 2
    cache_buckets.append(settings["cache_length"])
    metadata = {
        "config": source.config,
        "stages": {},
        "cache_length": settings["cache_length"],
        "cache_buckets": cache_buckets,
        "prefill_length": settings["prefill_length"],
        "predictor_cache_length": 16,
    }
    for phase, count in (("prefill", settings["prefill_length"]), ("decode", 1)):
        config = source.text
        for capacity in cache_buckets:
            cache = torch.zeros(
                1, config["num_key_value_heads"], capacity, config["head_dim"]
            )
            name = f"talker_{phase}_attention_{capacity}"
            metadata["stages"][name] = emit(
                name,
                source.attention(0),
                {
                    "hidden": torch.randn(1, count, config["hidden_size"]),
                    "keys": cache,
                    "values": cache.clone(),
                    "cache_positions": torch.arange(count),
                    "positions": torch.arange(count).repeat(3, 1),
                },
                "attention",
                {"weights.q", "weights.k", "weights.v", "weights.o"},
                args.output,
                args.compiler_build,
            )
        plans = [
            (
                f"talker_{phase}_router",
                source.router(0),
                {"hidden": torch.randn(count, config["hidden_size"])},
                "attention",
                set(),
            ),
            (
                f"talker_{phase}_shared",
                source.shared(0),
                {"hidden": torch.randn(count, config["hidden_size"])},
                "ffn",
                {"gate", "up", "down"},
            ),
            (
                f"talker_{phase}_norm",
                source.norm(),
                {"hidden": torch.randn(count, config["hidden_size"])},
                "attention",
                set(),
            ),
        ]
        for kind in ("text", "hidden"):
            plans.append(
                (
                    f"talker_{phase}_{kind}_projection",
                    source.resize(kind),
                    {
                        "hidden": torch.randn(
                            count, source.config["thinker_hidden_size"]
                        )
                    },
                    "ffn",
                    {"up", "down"},
                )
            )
        for name, stage, inputs, target, quantized in plans:
            metadata["stages"][name] = emit(
                name, stage, inputs, target, quantized, args.output, args.compiler_build
            )
            print(f"captured {name}", flush=True)
    for count in (1, 2, 4, 8, settings["prefill_length"]):
        name = f"talker_expert_{count}"
        metadata["stages"][name] = emit(
            name,
            source.expert(0, 0),
            {"hidden": torch.randn(count, source.text["hidden_size"])},
            "ffn",
            {"gate", "up", "down"},
            args.output,
            args.compiler_build,
        )
    metadata["stages"]["talker_output"] = emit(
        "talker_output",
        source.output(),
        {"hidden": torch.randn(1, source.text["hidden_size"])},
        "ffn",
        {"weight"},
        args.output,
        args.compiler_build,
    )
    for phase, count in (("prefill", 2), ("decode", 1)):
        config = source.predictor
        cache = torch.zeros(1, config["num_key_value_heads"], 16, config["head_dim"])
        for name, stage, inputs, target, quantized in (
            (
                f"predictor_{phase}_attention",
                source.attention(0, predictor=True),
                {
                    "hidden": torch.randn(1, count, config["hidden_size"]),
                    "keys": cache,
                    "values": cache.clone(),
                    "cache_positions": torch.arange(count),
                    "positions": torch.arange(count).repeat(3, 1),
                },
                "attention",
                {"weights.q", "weights.k", "weights.v", "weights.o"},
            ),
            (
                f"predictor_{phase}_dense",
                source.dense(0),
                {"hidden": torch.randn(count, config["hidden_size"])},
                "ffn",
                {"gate", "up", "down"},
            ),
            (
                f"predictor_{phase}_norm",
                source.norm(predictor=True),
                {"hidden": torch.randn(count, config["hidden_size"])},
                "attention",
                set(),
            ),
        ):
            metadata["stages"][name] = emit(
                name, stage, inputs, target, quantized, args.output, args.compiler_build
            )
            print(f"captured {name}", flush=True)
    metadata["stages"]["predictor_output"] = emit(
        "predictor_output",
        source.output(0),
        {"hidden": torch.randn(1, source.predictor["hidden_size"])},
        "ffn",
        {"weight"},
        args.output,
        args.compiler_build,
    )
    (args.output / "kernels.json").write_text(json.dumps(metadata, indent=2) + "\n")
    constants = {
        "width": source.text["hidden_size"],
        "thinkerWidth": source.config["thinker_hidden_size"],
        "layerCount": source.text["num_hidden_layers"],
        "expertCount": source.text["num_experts"],
        "topK": source.text["num_experts_per_tok"],
        "kvHeads": source.text["num_key_value_heads"],
        "headDim": source.text["head_dim"],
        "prefill": settings["prefill_length"],
        "capacity": settings["cache_length"],
        "vocabulary": source.text["vocab_size"],
        "predictorLayers": source.predictor["num_hidden_layers"],
        "predictorKVHeads": source.predictor["num_key_value_heads"],
        "predictorVocabulary": source.predictor["vocab_size"],
        "predictorCapacity": 16,
        "groups": source.predictor["num_code_groups"],
    }
    for kind, stage in (
        ("expert", "talker_expert_1"),
        ("shared", "talker_decode_shared"),
    ):
        for index, size in enumerate(metadata["stages"][stage]["weight_bytes"]):
            constants[f"{kind}Weight{index}"] = size
    declarations = ""
    for phase in ("prefill", "decode"):
        for capacity in cache_buckets:
            declarations += f'extern "C" void _mlir_ciface_forward_talker_{phase}_attention_{capacity}(AttentionResult *, Floats *, Bytes *, Hidden *, Cache *, Cache *, Slots *, Positions *);\n'
    header = (
        '#pragma once\n#include <cstddef>\n#include "view.h"\n'
        + declarations
        + "namespace TalkerParams {\n"
    )
    header += "".join(f"constexpr size_t {k} = {v};\n" for k, v in constants.items())
    header += (
        "constexpr size_t cacheBuckets[] = {"
        + ",".join(map(str, cache_buckets))
        + "};\n"
    )
    header += "using Attention = void (*)(AttentionResult *, Floats *, Bytes *, Hidden *, Cache *, Cache *, Slots *, Positions *);\n"
    for phase in ("prefill", "decode"):
        header += (
            f"constexpr Attention {phase}Attention[] = {{"
            + ",".join(
                f"_mlir_ciface_forward_talker_{phase}_attention_{capacity}"
                for capacity in cache_buckets
            )
            + "};\n"
        )
    (args.output / "talker-parameters.h").write_text(header + "}\n")


if __name__ == "__main__":
    main()
