import argparse
import json
from pathlib import Path
import sys

import torch

from .wave import SpanResidual, Weights

BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)


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
    metadata = {"config": config, "buckets": list(BUCKETS), "stages": {}, "steps": {}}
    for count in BUCKETS:
        hidden = torch.randn(count, config["hidden_size"])
        for kind, stage, inputs, quantized in (
            (
                "attention",
                source.attention(0),
                {"hidden": hidden, "positions": torch.arange(count)},
                {"weights.q", "weights.k", "weights.v", "weights.o"},
            ),
            (
                "dense",
                source.dense(0),
                {"hidden": hidden},
                {"weights.gate", "weights.up", "weights.down"},
            ),
            ("norm", source.norm(), {"hidden": hidden}, set()),
        ):
            name = f"wave_{kind}_{count}"
            metadata["stages"][name] = emit(
                name,
                stage,
                inputs,
                "attention" if kind != "dense" else "ffn",
                quantized,
                args.output,
                args.compiler_build,
            )
        length, channels = count, config["hidden_size"]
        steps = []

        def capture(kind, stage, output_length, output_channels, *, span=False):
            nonlocal length, channels
            name = f"wave_{kind}_{count}"
            owned_rows = ((length + 7) // 8 + 15) // 16 * 16 if span else 0
            if span:
                stage = SpanResidual(stage, owned_rows).eval()
            halo = stage.halo if span else 0
            quantized = {
                key
                for key, _ in stage.named_parameters()
                if key.endswith("weight") or key in ("weights.up", "weights.down")
            }
            # Normalization weights are FP32; only convolutions and pointwise projections use MXFP8.
            quantized.discard("weights.norm")
            spec = emit(
                name,
                stage,
                {
                    "hidden": torch.randn(
                        1, channels, owned_rows + halo if span else length
                    )
                },
                "ffn",
                quantized,
                args.output,
                args.compiler_build,
                local=span,
            )
            metadata["stages"][name] = spec
            steps.append(
                {
                    "name": name,
                    "kind": kind,
                    "input": [channels, length],
                    "output": [output_channels, output_length],
                    "owned_rows": owned_rows,
                    "halo": halo,
                }
            )
            length, channels = output_length, output_channels

        for index, factor in enumerate(config["upsampling_ratios"]):
            capture(f"up{index}", source.upsample(index), length * factor, channels)
            capture(f"next{index}", source.convnext(index), length, channels)
        capture("input", source.decoder_input(), length, config["decoder_dim"])
        for index, factor in enumerate(config["upsample_rates"]):
            capture(
                f"decoder{index}_up",
                source.decoder_upsample(index),
                (length - 1) * factor,
                channels // 2,
            )
            for unit in range(3):
                capture(
                    f"decoder{index}_res{unit}",
                    source.residual(index, unit),
                    length,
                    channels,
                    span=True,
                )
        capture("output", source.output(), length, 1)
        metadata["steps"][str(count)] = steps
        print(f"captured Code2Wav bucket {count}, output samples {length}", flush=True)
    (args.output / "kernels.json").write_text(json.dumps(metadata, indent=2) + "\n")
    declarations = []
    for count in BUCKETS:
        declarations.extend(
            [
                f'extern "C" void _mlir_ciface_forward_wave_attention_{count}'
                "(Matrix *, Floats *, Bytes *, Matrix *, Slots *);",
                f'extern "C" void _mlir_ciface_forward_wave_dense_{count}(Matrix *, Floats *, Bytes *, Matrix *);',
                f'extern "C" void _mlir_ciface_forward_wave_norm_{count}(Matrix *, Floats *, Matrix *);',
            ]
        )
        declarations.extend(
            f'extern "C" void _mlir_ciface_forward_{step["name"]}(Hidden *, Floats *, Bytes *, Hidden *);'
            for step in metadata["steps"][str(count)]
        )
    header = (
        "#pragma once\n#include <cstddef>\n"
        + "\n".join(declarations)
        + "\nnamespace WaveParams {\n"
    )
    constants = {
        "width": config["hidden_size"],
        "layers": config["num_hidden_layers"],
        "groups": config["num_quantizers"],
        "vocabulary": config["codebook_size"],
        "stepCount": len(metadata["steps"]["1"]),
        "totalUpsample": 1920,
        "tail": 555,
    }
    header += "".join(
        f"constexpr size_t {name} = {value};\n" for name, value in constants.items()
    )
    header += "constexpr size_t buckets[] = {" + ",".join(map(str, BUCKETS)) + "};\n"
    header += (
        "using Attention = void (*)(Matrix *, Floats *, Bytes *, Matrix *, Slots *);\n"
    )
    header += "using Dense = void (*)(Matrix *, Floats *, Bytes *, Matrix *);\n"
    header += "using Norm = void (*)(Matrix *, Floats *, Matrix *);\n"
    header += "using Convolution = void (*)(Hidden *, Floats *, Bytes *, Hidden *);\n"
    for kind, typedef in (
        ("attention", "Attention"),
        ("dense", "Dense"),
        ("norm", "Norm"),
    ):
        header += (
            f"constexpr {typedef} {kind}[] = {{"
            + ",".join(f"_mlir_ciface_forward_wave_{kind}_{count}" for count in BUCKETS)
            + "};\n"
        )
    header += "struct Step { size_t inputChannels, inputLength, outputChannels, outputLength; Convolution run; size_t ownedRows, halo; };\n"
    header += "constexpr Step steps[][stepCount] = {\n"
    for count in BUCKETS:
        entries = []
        for step in metadata["steps"][str(count)]:
            values = [*step["input"], *step["output"]]
            entries.append(
                "{"
                + ",".join(map(str, values))
                + f',_mlir_ciface_forward_{step["name"]}'
                + f',{step["owned_rows"]},{step["halo"]}'
                + "}"
            )
        header += "{" + ",".join(entries) + "},\n"
    header += "};\n}\n"
    (args.output / "wave-parameters.h").write_text(header)


if __name__ == "__main__":
    main()
