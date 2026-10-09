import argparse
import json
from pathlib import Path
import struct
import sys
import tomllib

import torch

from ..trace.talker import Weights


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

    torch.set_num_threads(1)
    source = Weights(Path(json.loads(args.checkpoint_config.read_text())["path"]))
    metadata = json.loads(args.kernels.read_text())
    stages = metadata["stages"]
    attention_name = f"talker_prefill_attention_{metadata['cache_length']}"
    for capacity in metadata["cache_buckets"]:
        for phase in ("prefill", "decode"):
            if stages[f"talker_{phase}_attention_{capacity}"] != stages[attention_name]:
                raise ValueError("Talker attention bucket parameter contracts differ")
    for kind in (
        "router",
        "shared",
        "norm",
        "text_projection",
        "hidden_projection",
    ):
        if stages[f"talker_prefill_{kind}"] != stages[f"talker_decode_{kind}"]:
            raise ValueError(f"Talker {kind} parameter contracts differ")
    for count in (2, 4, 8, metadata["prefill_length"]):
        if stages[f"talker_expert_{count}"] != stages["talker_expert_1"]:
            raise ValueError("Talker expert bucket parameter contracts differ")
    for kind in ("attention", "dense", "norm"):
        if stages[f"predictor_prefill_{kind}"] != stages[f"predictor_decode_{kind}"]:
            raise ValueError(f"code predictor {kind} parameter contracts differ")
    system = tomllib.loads(args.system_config.read_text())
    (chip,) = [c for c in system["chips"] if c["role"] == "media"]
    tile = system["media"]["talker_tile"]
    directory = args.output / f"chip-{chip['id']}" / f"tile-{tile}"
    directory.mkdir(parents=True, exist_ok=True)
    regions = []
    with (directory / "talker.f32").open("wb") as floats, (
        directory / "talker.bin"
    ).open("wb") as packed:
        for group in range(source.config["num_code_groups"]):
            name = (
                "talker.model.codec_embedding.weight"
                if group == 0
                else f"talker.code_predictor.model.codec_embedding.{group - 1}.weight"
            )
            tensor = source.tensor(name)
            vocabulary = (
                source.text["vocab_size"]
                if group == 0
                else source.predictor["vocab_size"]
            )
            if list(tensor.shape) != [vocabulary, source.text["hidden_size"]]:
                raise ValueError(
                    "codec embedding shape differs from model configuration"
                )
            rows = pack_rows(tensor.detach())
            regions.append((floats.tell() // 4, 0, packed.tell(), rows.numel()))
            packed.write(rows.cpu().contiguous().numpy().tobytes())

        def append(stage, name):
            spec = stages[name]
            values = dict(stage.named_parameters()) | dict(stage.named_buffers())
            first_float, first_byte = floats.tell(), packed.tell()
            for key, shape in zip(spec["parameters"], spec["shapes"], strict=True):
                tensor = values[key]
                if list(tensor.shape) != shape:
                    raise ValueError(f"Talker parameter shape differs: {key}")
                array = tensor.detach()
                if key in spec["embeddings"]:
                    packed.write(pack_rows(array).cpu().contiguous().numpy().tobytes())
                elif key in spec["quantized"]:
                    packed.write(
                        pack_matrix(*quantize(array), spec["layouts"][key])
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
                    "Talker packed parameter sizes differ from compiled kernel"
                )
            regions.append(region)

        append(source.resize("text"), "talker_prefill_text_projection")
        append(source.resize("hidden"), "talker_prefill_hidden_projection")
        append(source.norm(), "talker_prefill_norm")
        append(source.output(), "talker_output")
        append(source.norm(predictor=True), "predictor_prefill_norm")
        for group in range(source.config["num_code_groups"] - 1):
            append(source.output(group), "predictor_output")
        for layer in range(source.text["num_hidden_layers"]):
            append(source.attention(layer), attention_name)
            append(source.router(layer), "talker_prefill_router")
            append(source.shared(layer), "talker_prefill_shared")
            for expert in range(source.text["num_experts"]):
                append(source.expert(layer, expert), "talker_expert_1")
            print(f"packed Talker layer {layer}", flush=True)
        for layer in range(source.predictor["num_hidden_layers"]):
            append(
                source.attention(layer, predictor=True), "predictor_prefill_attention"
            )
            append(source.dense(layer), "predictor_prefill_dense")
        sizes = (floats.tell() // 4, packed.tell())
    with (directory / "talker-layout.bin").open("wb") as layout:
        layout.write(struct.pack("<4Q", 0x54414C4B0001, *sizes, len(regions)))
        for region in regions:
            layout.write(struct.pack("<4Q", *region))
    placement = {
        "chip": chip["id"],
        "tile": tile,
        "float_elements": sizes[0],
        "weight_bytes": sizes[1],
        "embedding_format": "mxfp8_rows",
    }
    (args.output / "talker-placement.json").write_text(
        json.dumps(placement, indent=2) + "\n"
    )
    print(f"Talker weights: {(sizes[0] * 4 + sizes[1]) / 2**30:.3f} GiB", flush=True)


if __name__ == "__main__":
    main()
