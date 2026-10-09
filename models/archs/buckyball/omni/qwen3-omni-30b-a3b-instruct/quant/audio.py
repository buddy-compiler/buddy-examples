import argparse
import json
from pathlib import Path
import struct
import sys
import tomllib

from ..trace.audio import Weights


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
    from stack.compiler.quant.mxfp8_embedding import pack_rows
    from examples.balls.mxmm.compiler.python.layout import pack as pack_matrix

    metadata = json.loads(args.kernels.read_text())
    system = tomllib.loads(args.system_config.read_text())
    (chip,) = [c for c in system["chips"] if c["role"] == "media"]
    tile = system["media"]["audio_tile"]
    source = Weights(Path(json.loads(args.checkpoint_config.read_text())["path"]))
    directory = args.output / f"chip-{chip['id']}" / f"tile-{tile}"
    directory.mkdir(parents=True, exist_ok=True)
    for kind in ("attention", "mlp", "output"):
        spec = metadata["stages"][f"audio_{kind}_1"]
        if any(
            metadata["stages"][f"audio_{kind}_{n}"] != spec for n in metadata["buckets"]
        ):
            raise ValueError(f"audio {kind} bucket parameter contracts differ")
    jobs = [(f"audio_conv{i}", source.convolution(i)) for i in range(1, 4)]
    jobs.append(("audio_projection", source.projection()))
    for layer in range(source.config["encoder_layers"]):
        jobs.extend(
            [
                ("audio_attention_1", source.attention(layer)),
                ("audio_mlp_1", source.mlp(layer)),
            ]
        )
    jobs.append(("audio_output_1", source.output()))
    regions = []
    with (directory / "audio.f32").open("wb") as floats, (directory / "audio.bin").open(
        "wb"
    ) as packed:
        for name, stage in jobs:
            spec = metadata["stages"][name]
            values = dict(stage.named_parameters()) | dict(stage.named_buffers())
            first_float, first_byte = floats.tell(), packed.tell()
            for name, shape in zip(spec["parameters"], spec["shapes"], strict=True):
                tensor = values[name]
                if list(tensor.shape) != shape:
                    raise ValueError(f"audio parameter shape differs: {name}")
                array = tensor.detach()
                if name in spec["embeddings"]:
                    packed.write(pack_rows(array).cpu().contiguous().numpy().tobytes())
                elif name in spec["quantized"]:
                    packed.write(
                        pack_matrix(*quantize(array), spec["layouts"][name])
                        .cpu()
                        .contiguous()
                        .numpy()
                        .tobytes()
                    )
                else:
                    floats.write(array.cpu().contiguous().numpy().tobytes())
            region = (
                first_float // 4,
                (floats.tell() - first_float) // 4,
                first_byte,
                packed.tell() - first_byte,
            )
            if region[1] != spec["floats"] or region[3] != spec["bytes"]:
                raise ValueError(
                    "audio packed parameter sizes differ from compiled kernel"
                )
            regions.append(region)
        sizes = (floats.tell() // 4, packed.tell())
    with (directory / "audio-layout.bin").open("wb") as layout:
        layout.write(struct.pack("<4Q", 0x415544490001, *sizes, len(regions)))
        for region in regions:
            layout.write(struct.pack("<4Q", *region))
    result = {
        "chip": chip["id"],
        "tile": tile,
        "float_elements": sizes[0],
        "weight_bytes": sizes[1],
    }
    (args.output / "audio-placement.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(f"audio weights: {(sizes[0] * 4 + sizes[1]) / 2**30:.3f} GiB", flush=True)


if __name__ == "__main__":
    main()
