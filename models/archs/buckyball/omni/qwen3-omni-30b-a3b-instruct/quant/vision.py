import argparse
import json
from pathlib import Path
import struct
import sys
import tomllib

from ..trace.vision import Weights


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

    metadata = json.loads(args.kernels.read_text())
    system = tomllib.loads(args.system_config.read_text())
    (chip,) = [c for c in system["chips"] if c["role"] == "media"]
    tile = system["media"]["vision_tile"]
    source = Weights(Path(json.loads(args.checkpoint_config.read_text())["path"]))
    directory = args.output / f"chip-{chip['id']}" / f"tile-{tile}"
    directory.mkdir(parents=True, exist_ok=True)
    count = metadata["buckets"][0]
    for kind in ("patch", "attention", "mlp", "merge", "deepstack"):
        specification = metadata["stages"][f"vision_{kind}_{count}"]
        if any(
            metadata["stages"][f"vision_{kind}_{n}"] != specification
            for n in metadata["buckets"]
        ):
            raise ValueError(f"vision {kind} bucket parameter contracts differ")
    jobs = [("patch", source.patch())]
    for layer in range(source.config["depth"]):
        jobs.extend(
            [("attention", source.attention(layer)), ("mlp", source.mlp(layer))]
        )
    jobs.append(("merge", source.merger(-1)))
    jobs.extend(
        ("deepstack", source.merger(i))
        for i in range(len(source.config["deepstack_visual_indexes"]))
    )
    regions = []
    with (directory / "vision.f32").open("wb") as floats, (
        directory / "vision.bin"
    ).open("wb") as packed:
        for kind, stage in jobs:
            spec = metadata["stages"][f"vision_{kind}_{count}"]
            values = dict(stage.named_parameters()) | dict(stage.named_buffers())
            first_float, first_byte = floats.tell(), packed.tell()
            for name, shape in zip(spec["parameters"], spec["shapes"], strict=True):
                tensor = values[name]
                if list(tensor.shape) != shape:
                    raise ValueError(f"vision parameter shape differs: {name}")
                array = tensor.detach().numpy()
                if name in spec["quantized"]:
                    packed.write(pack_matrix(*quantize(array), spec["layouts"][name]).tobytes())
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
                    "vision packed parameter sizes differ from compiled kernel"
                )
            regions.append(region)
        sizes = (floats.tell() // 4, packed.tell())
    with (directory / "vision-layout.bin").open("wb") as layout:
        layout.write(struct.pack("<4Q", 0x564953490001, *sizes, len(regions)))
        for region in regions:
            layout.write(struct.pack("<4Q", *region))
    result = {
        "chip": chip["id"],
        "tile": tile,
        "float_elements": sizes[0],
        "weight_bytes": sizes[1],
    }
    (args.output / "vision-placement.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(f"vision weights: {(sizes[0] * 4 + sizes[1]) / 2**30:.3f} GiB", flush=True)


if __name__ == "__main__":
    main()
