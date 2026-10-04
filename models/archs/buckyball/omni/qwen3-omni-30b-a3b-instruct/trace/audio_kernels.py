import argparse
import json
from pathlib import Path
import sys

import torch

from .audio import Weights

BUCKETS = (1, 4, 16, 64, 128)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-config", type=Path, required=True)
    parser.add_argument("--compiler-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.compiler_build / "python_packages"))
    from .graph import emit

    source = Weights(Path(json.loads(args.checkpoint_config.read_text())["path"]))
    config = source.config
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    metadata = {"config": config, "buckets": list(BUCKETS), "stages": {}}
    shapes = [(1, 1, 128, 100), (1, 480, 64, 50), (1, 480, 32, 25)]
    for index, shape in enumerate(shapes, 1):
        name = f"audio_conv{index}"
        metadata["stages"][name] = emit(
            name,
            source.convolution(index),
            {
                "features": torch.randn(shape),
                "frames": torch.tensor([shape[-1]], dtype=torch.int64),
            },
            "attention",
            {"weight"},
            args.output,
            args.compiler_build,
        )
        print(f"captured {name}", flush=True)
    name = "audio_projection"
    metadata["stages"][name] = emit(
        name,
        source.projection(),
        {"features": torch.randn(1, 480, 16, 13)},
        "attention",
        {"weight"},
        args.output,
        args.compiler_build,
    )
    for count in BUCKETS:
        hidden = torch.randn(count, config["d_model"])
        for kind, stage, inputs, target, quantized in (
            (
                "attention",
                source.attention(0),
                {"hidden": hidden, "mask": torch.zeros(count, count, dtype=torch.bool)},
                "attention",
                {"weights.qkv", "weights.projection"},
            ),
            (
                "mlp",
                source.mlp(0),
                {"hidden": hidden},
                "ffn",
                {"weights.up", "weights.down"},
            ),
            (
                "output",
                source.output(),
                {"hidden": hidden},
                "ffn",
                {"weights.up", "weights.down"},
            ),
        ):
            name = f"audio_{kind}_{count}"
            metadata["stages"][name] = emit(
                name, stage, inputs, target, quantized, args.output, args.compiler_build
            )
            print(f"captured {name}", flush=True)
    (args.output / "kernels.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "audio-parameters.h").write_text(
        "#pragma once\n#include <cstddef>\n"
        + f"constexpr size_t audioWidth = {config['d_model']};\n"
        + f"constexpr size_t audioLayers = {config['encoder_layers']};\n"
        + f"constexpr size_t audioOutputWidth = {config['output_dim']};\n"
        + "constexpr size_t audioBuckets[] = {"
        + ",".join(map(str, BUCKETS))
        + "};\n"
    )


if __name__ == "__main__":
    main()
