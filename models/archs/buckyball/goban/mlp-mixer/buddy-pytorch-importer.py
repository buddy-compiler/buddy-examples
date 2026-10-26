from stack.compiler.frontend.trace import trace
import numpy as np
from PIL import Image
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import argparse
from importlib import import_module
import json
import hashlib
from pathlib import Path

import torch
from torch._inductor.decomposition import decompositions as inductor_decomp


def main():
    parser = argparse.ArgumentParser(description="MLP-Mixer AOT importer")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compiler-build", type=Path, required=True)
    parser.add_argument("--design-pb", type=Path, required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--revision", type=str, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    Mixer = import_module("stack.models.models.mlp-mixer.model").Mixer
    compiler = args.compiler_build.resolve()
    from buddy.compiler.frontend import DynamoCompiler
    from buddy.compiler.graph.transform import simply_fuse
    from stack.compiler.frontend.ops import tosa

    path = hf_hub_download(args.checkpoint, "model.safetensors", revision=args.revision)
    state = load_file(path)
    # The non-overlapping 16x16 patch projection is a linear map. Flatten
    # its C,H,W weight dimensions to match the prepared patch input layout.
    state["stem.proj.weight"] = state["stem.proj.weight"].flatten(1)
    model = Mixer().eval()
    model.load_state_dict(state, strict=True)
    image = Image.open(
        args.repo_root / "stack/models/models/alexnet/images/dog.bmp"
    ).convert("RGB")
    width, height = image.size
    size = 256
    target = (
        (size, int(size * height / width))
        if width <= height
        else (int(size * width / height), size)
    )
    image = image.resize(target, Image.Resampling.BICUBIC)
    left, top = round((image.width - 224) / 2), round((image.height - 224) / 2)
    image = image.crop((left, top, left + 224, top + 224))
    pixels = np.asarray(image, dtype=np.float32) / np.float32(127.5) - np.float32(1.0)
    pixels = torch.from_numpy(pixels.transpose(2, 0, 1).copy())[None]
    data = (
        pixels.reshape(1, 3, 14, 16, 14, 16)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(196, 768)
        .contiguous()
    )
    import __init__ as recipe
    from stack.compiler.frontend.quant.importer import quantize_model_graph
    from stack.compiler.frontend.quant.offline.activations import calibrate_layer_ranges
    from quant.calibration import ImageNetCalibration
    import tomllib

    config = tomllib.loads((Path(__file__).parent / "quant/quant.toml").read_text())
    calibration = calibrate_layer_ranges(
        model, ImageNetCalibration(), batch_size=None, **config["calibration"]
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    decompositions = dict(inductor_decomp)
    importer = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=decompositions,
        trace=None,
    )
    with torch.no_grad():
        graphs = importer.importer(model, data)
    if len(graphs) != 1:
        raise ValueError(f"MLP-Mixer must import as one graph, got {len(graphs)}")
    graph = graphs[0]
    params = importer.imported_params[graph]
    parameter_names = {id(value): name for name, value in model.named_parameters()}
    names = [parameter_names[id(value)] for value in params]
    (output / "graph.json").write_text(
        json.dumps(
            [
                {
                    "name": node.name,
                    "op": type(node).__name__,
                    "args": node.args,
                    "shape": node.tensor_meta.get("shape"),
                }
                for node in graph.body
            ],
            default=str,
            indent=2,
        )
    )
    quantize_model_graph(
        graph,
        params,
        names,
        output,
        "mixer",
        calibration,
    )
    graph.dump(output / "03-quanted.txt")
    graph.fuse_ops([simply_fuse])
    graph.dump(output / "04-fusion.txt")
    trace(graph, Path(__file__).parent / "trace/trace.toml")
    graph.dump(output / "05-traced.txt")
    (output / "quant-graph.json").write_text(
        json.dumps(
            [
                {
                    "name": node.name,
                    "op": type(node).__name__,
                    "args": node.args,
                    "parents": node._parents,
                    "shape": node.tensor_meta.get("shape"),
                }
                for node in graph.body
            ],
            default=str,
            indent=2,
        )
    )
    from buddy.compiler.graph import GraphDriver

    driver = GraphDriver(graph)
    (subgraph,) = driver.subgraphs
    subgraph.lower_to_top_level_ir(buffer_layout="contiguous")
    (output / "subgraph0.mlir").write_text(str(subgraph._imported_module))
    (output / "forward.mlir").write_text(str(driver.construct_main_graph(True)))
    data.numpy().tofile(output / "input.f32")
    metadata = {
        "version": 1,
        "design": args.design,
        "model": "mlp-mixer",
        "checkpoint": args.checkpoint,
        "revision": args.revision,
        "input_shape": [196, 768],
        "input_sha256": hashlib.sha256(data.numpy().tobytes()).hexdigest(),
        "compute_cores": 4,
        "weight_dtype": "symmetric_int8_per_output_channel",
        "accumulator_dtype": "int32",
    }
    (output / "model.json").write_text(json.dumps(metadata))

    (output / "params.f32").replace(output / "arg0.data")
    (output / "weights.bin").replace(output / "arg1.data")


if __name__ == "__main__":
    main()
