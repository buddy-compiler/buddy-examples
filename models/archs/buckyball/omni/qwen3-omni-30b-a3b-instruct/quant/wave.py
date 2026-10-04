import argparse
import json
from pathlib import Path
import struct
import sys
import tomllib

import torch

from ..trace.wave import Weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-config", type=Path, required=True)
    parser.add_argument("--compiler-build", type=Path, required=True)
    parser.add_argument("--kernels", type=Path, required=True)
    parser.add_argument("--system-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.compiler_build / "python_packages"))
    from buddy.compiler.graph.transform.quantization.mxfp8 import quantize
    from examples.balls.mxmm.compiler.python.layout import pack as pack_matrix

    torch.set_num_threads(1)
    source = Weights(Path(json.loads(args.checkpoint_config.read_text())["path"]))
    metadata = json.loads(args.kernels.read_text())
    stages = metadata["stages"]
    for kind in ("attention", "dense", "norm"):
        for count in metadata["buckets"]:
            if stages[f"wave_{kind}_{count}"] != stages[f"wave_{kind}_1"]:
                raise ValueError(f"Code2Wav {kind} parameter contracts differ")
    for index, step in enumerate(metadata["steps"]["1"]):
        for count in metadata["buckets"]:
            other = metadata["steps"][str(count)][index]
            if (
                step["kind"] != other["kind"]
                or stages[step["name"]] != stages[other["name"]]
            ):
                raise ValueError(
                    "Code2Wav convolution bucket parameter contracts differ"
                )
    system = tomllib.loads(args.system_config.read_text())
    (chip,) = [c for c in system["chips"] if c["role"] == "media"]
    tile = system["media"]["code2wav_tile"]
    directory = args.output / f"chip-{chip['id']}" / f"tile-{tile}"
    directory.mkdir(parents=True, exist_ok=True)
    regions = []
    with (directory / "wave.f32").open("wb") as floats, (directory / "wave.bin").open(
        "wb"
    ) as packed:
        embedding = source.tensor("code2wav.code_embedding.weight")
        regions.append((0, embedding.numel(), 0, 0))
        floats.write(embedding.numpy().tobytes())

        def append(stage, name):
            spec = stages[name]
            values = dict(stage.named_parameters()) | dict(stage.named_buffers())
            first_float, first_byte = floats.tell(), packed.tell()
            for key, shape in zip(spec["parameters"], spec["shapes"], strict=True):
                tensor = values[key]
                if list(tensor.shape) != shape:
                    raise ValueError(f"Code2Wav parameter shape differs: {key}")
                array = tensor.detach().numpy()
                if key in spec["quantized"]:
                    packed.write(pack_matrix(*quantize(array), spec["layouts"][key]).tobytes())
                else:
                    floats.write(array.tobytes())
            region = (
                first_float // 4,
                (floats.tell() - first_float) // 4,
                first_byte,
                packed.tell() - first_byte,
            )
            if region[1] != spec["floats"] or region[3] != spec["bytes"]:
                raise ValueError(
                    "Code2Wav packed parameter sizes differ from compiled kernel"
                )
            regions.append(region)

        for layer in range(source.config["num_hidden_layers"]):
            append(source.attention(layer), "wave_attention_1")
            append(source.dense(layer), "wave_dense_1")
        append(source.norm(), "wave_norm_1")
        for index in range(len(source.config["upsampling_ratios"])):
            append(source.upsample(index), f"wave_up{index}_1")
            append(source.convnext(index), f"wave_next{index}_1")
        append(source.decoder_input(), "wave_input_1")
        for index in range(len(source.config["upsample_rates"])):
            append(source.decoder_upsample(index), f"wave_decoder{index}_up_1")
            for unit in range(3):
                append(source.residual(index, unit), f"wave_decoder{index}_res{unit}_1")
        append(source.output(), "wave_output_1")
        sizes = (floats.tell() // 4, packed.tell())
    with (directory / "wave-layout.bin").open("wb") as layout:
        layout.write(struct.pack("<4Q", 0x574156450001, *sizes, len(regions)))
        for region in regions:
            layout.write(struct.pack("<4Q", *region))
    placement = {
        "chip": chip["id"],
        "tile": tile,
        "float_elements": sizes[0],
        "weight_bytes": sizes[1],
    }
    (args.output / "wave-placement.json").write_text(
        json.dumps(placement, indent=2) + "\n"
    )
    print(f"Code2Wav weights: {(sizes[0] * 4 + sizes[1]) / 2**30:.3f} GiB", flush=True)


if __name__ == "__main__":
    main()
