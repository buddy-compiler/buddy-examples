import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import struct
import sys

import torch

from ..trace.moe import Weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-config", type=Path, required=True)
    parser.add_argument("--compiler-build", type=Path, required=True)
    parser.add_argument("--kernels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.checkpoint = Path(json.loads(args.checkpoint_config.read_text())["path"])
    sys.path.insert(0, str(args.compiler_build / "python_packages"))
    from buddy.compiler.graph.transform.quantization.mxfp8 import quantize
    from ..permute.layout import reorder

    metadata = json.loads(args.kernels.read_text())
    stages = metadata["stages"]
    for kind in ("attention", "router", "embedding", "norm"):
        if stages[f"prefill_{kind}"] != stages[f"decode_{kind}"]:
            raise ValueError(f"{kind} parameter order differs between prefill and decode")
    chips = [chip for chip in metadata["system"]["chips"] if chip["role"] == "thinker"]
    tiles = metadata["system"]["thinker"]["tensor_tiles"]
    torch.set_num_threads(1)

    def pack(job):
        chip, tile, rank = job
        source = Weights(args.checkpoint)
        directory = args.output / f"chip-{chip['id']}" / f"tile-{tile}"
        directory.mkdir(parents=True, exist_ok=True)
        begin, end = chip["layers"]
        ranges = []
        with (directory / "params.f32").open("wb") as floats, (directory / "weights.bin").open("wb") as packed:
            def append(stage, specification):
                values = dict(stage.named_parameters()) | dict(stage.named_buffers())
                first_float, first_byte = floats.tell(), packed.tell()
                for name, shape in zip(specification["parameters"], specification["shapes"]):
                    tensor = values[name]
                    if list(tensor.shape) != shape or tensor.dtype != torch.float32:
                        raise ValueError(f"parameter contract mismatch: {name}")
                    array = tensor.detach().numpy()
                    if name in specification["quantized"]:
                        packed.write(reorder(*quantize(array)).tobytes())
                    else:
                        floats.write(array.tobytes())
                count_float = (floats.tell() - first_float) // 4
                count_byte = packed.tell() - first_byte
                if count_float != specification["floats"] or count_byte != specification["bytes"]:
                    raise ValueError("packed parameter sizes differ from compiled kernel")
                ranges.append((first_float // 4, count_float, first_byte, count_byte))

            if begin == 0:
                append(source.embedding(rank, len(tiles)), stages["prefill_embedding"])
            else:
                ranges.append((0, 0, 0, 0))
            if end == source.config["num_hidden_layers"]:
                append(source.output(rank, len(tiles)), stages["output"])
            else:
                ranges.append((0, 0, 0, 0))
            if end == source.config["num_hidden_layers"]:
                append(source.norm(), stages["prefill_norm"])
            else:
                ranges.append((0, 0, 0, 0))
            for layer in range(begin, end):
                append(source.attention(layer, rank, len(tiles)), stages["prefill_attention"])
                append(source.router(layer), stages["prefill_router"])
                for expert in range(source.config["num_experts"]):
                    append(source.expert(layer, expert, rank, len(tiles)), stages["expert"])
                print(f"chip {chip['id']} tile {tile}: packed layer {layer}", flush=True)
            counts = (floats.tell() // 4, packed.tell())
        with (directory / "layout.bin").open("wb") as output:
            output.write(struct.pack("<8Q", 0x4F4D4E490001, begin, end, rank, len(tiles), *counts, len(ranges)))
            for region in ranges:
                output.write(struct.pack("<4Q", *region))
        return {"chip": chip["id"], "tile": tile, "layers": [begin, end], "tensor_rank": rank,
                "float_elements": counts[0], "weight_bytes": counts[1]}

    jobs = [(chip, tile, rank) for chip in chips for rank, tile in enumerate(tiles)]
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        placements = list(pool.map(pack, jobs))
    for chip in chips:
        used = sum(p["float_elements"] * 4 + p["weight_bytes"] for p in placements if p["chip"] == chip["id"])
        if used >= chip["memory_mib"] << 20:
            raise ValueError(f"chip {chip['id']} weights exceed its DDR")
        print(f"chip {chip['id']}: resident Thinker weights {used / 2**30:.3f} GiB", flush=True)
    (args.output / "placements.json").write_text(json.dumps(placements, indent=2) + "\n")


if __name__ == "__main__":
    main()
