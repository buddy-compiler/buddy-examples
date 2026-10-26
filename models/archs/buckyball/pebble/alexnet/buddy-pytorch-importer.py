from stack.compiler.frontend.trace import trace
import numpy as np
from PIL import Image
from torchvision.models import AlexNet_Weights, alexnet
import argparse
from importlib import import_module
import json
import tomllib
from math import prod
from pathlib import Path

import torch
from torch._decomp import remove_decompositions
from torch._inductor.decomposition import decompositions as inductor_decomp


def main():
    parser = argparse.ArgumentParser(description="AlexNet AOT importer")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compiler-build", type=Path, required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    compiler = args.compiler_build.resolve()
    from buddy.compiler.frontend import DynamoCompiler
    from buddy.compiler.graph import GraphDriver
    from buddy.compiler.graph.transform import simply_fuse
    from examples.designs.pebble.stack.midend import ops as pebble

    # ===----------------------------------------------------------------------===
    # Step 1: Load the trained model and prepare the graph capture input.
    # ===----------------------------------------------------------------------===
    model = alexnet(weights=None).eval()
    model.load_state_dict(
        torch.load(
            args.repo_root / "stack/models/models/alexnet/output/alexnet/model.pth",
            weights_only=True,
        )
    )
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False
    pixels = np.asarray(
        Image.open(
            args.repo_root / "stack/models/models/alexnet/images/dog.bmp"
        ).convert("RGB"),
        dtype=np.float32,
    )
    if pixels.shape != (224, 224, 3):
        raise ValueError(f"AlexNet input must be 224x224 RGB, got {pixels.shape}")
    value = pixels / np.float32(255.0)
    value = (value - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    data = torch.from_numpy(value.transpose(2, 0, 1).copy())[None]
    from quant.calibration import ImageNetCalibration
    from stack.compiler.frontend.quant.offline.activations import calibrate_layer_ranges
    from stack.compiler.frontend.quant.offline.weights import quantize_weights
    from stack.compiler.frontend.quant.importer import quantize_model_graph

    # ===----------------------------------------------------------------------===
    # Step 1.1: Calibrate offline activation scales.
    # ===----------------------------------------------------------------------===
    config = tomllib.loads((Path(__file__).parent / "quant/quant.toml").read_text())
    if config["activations"]["scale_mode"] != "offline":
        raise ValueError("Pebble requires offline activation scales")
    weights = quantize_weights(model.state_dict(), config["weights"])
    calibration = calibrate_layer_ranges(
        model, ImageNetCalibration(), quantized=weights, **config["calibration"]
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    # ===----------------------------------------------------------------------===
    # Step 2: Import the model with the configured trace nodes.
    # ===----------------------------------------------------------------------===

    decompositions = dict(inductor_decomp)
    remove_decompositions(
        decompositions, [torch.ops.aten.max_pool2d_with_indices.default]
    )
    importer = DynamoCompiler(
        primary_registry=pebble.ops_registry,
        aot_autograd_decomposition=decompositions,
        trace=None,
    )
    with torch.no_grad():
        graphs = importer.importer(model, data)
    if len(graphs) != 1:
        raise ValueError(f"AlexNet must import as one graph, got {len(graphs)}")
    graph = graphs[0]

    # ===----------------------------------------------------------------------===
    # Step 2.1: Export the imported graph.
    # ===----------------------------------------------------------------------===
    params = importer.imported_params[graph]
    graph.dump(output / "01-import.txt")
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

    # ===----------------------------------------------------------------------===
    # Step 2.2: Quantize and lower the graph using the target weight layout.
    # ===----------------------------------------------------------------------===
    quantize_model_graph(
        graph,
        params,
        names,
        output,
        "alexnet",
        calibration,
        weight_scales={
            name: scale.numpy() for name, scale in weights["scales"].items()
        },
    )
    graph.dump(output / "03-quanted.txt")
    graph.fuse_ops([simply_fuse])
    graph.dump(output / "04-fusion.txt")
    trace(graph, Path(__file__).parent / "trace/trace.toml")
    graph.dump(output / "05-traced.txt")
    graph.dump(output / "02-quanted.txt")

    # ===----------------------------------------------------------------------===
    # Step 3: Export the target subgraphs and entry point to MLIR.
    # ===----------------------------------------------------------------------===
    driver = GraphDriver(graph)
    for index, subgraph in enumerate(driver.subgraphs):
        subgraph.lower_to_top_level_ir(buffer_layout="contiguous")
        # Design layouts are already contiguous in the packed payload. Retain
        # their strides while allowing nonzero offsets into that payload.
        (output / f"subgraph{index}.mlir").write_text(str(subgraph._imported_module))
    (output / "forward.mlir").write_text(str(driver.construct_main_graph(True)))

    # ===----------------------------------------------------------------------===
    # Step 4: Export model metadata.
    # ===----------------------------------------------------------------------===
    metadata = {
        "version": 1,
        "design": args.design,
        "model": "alexnet",
        "checkpoint": "models/models/alexnet/output/alexnet/model.pth",
        "input_shape": [1, 3, 224, 224],
    }
    (output / "model.json").write_text(json.dumps(metadata))


if __name__ == "__main__":
    main()
