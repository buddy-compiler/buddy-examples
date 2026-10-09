import argparse
from importlib import import_module
import json
from math import prod
from pathlib import Path
import sys

import torch
from torch._decomp import remove_decompositions
from torch._inductor.decomposition import decompositions as inductor_decomp

from .prepare import load_input, load_model


def main():
    parser = argparse.ArgumentParser(description="AlexNet AOT importer")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compiler-build", type=Path, required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--chip", required=True)
    parser.add_argument("--trace", action="store_true")
    import_module(".configs.importer-param", __package__).add_arguments(parser)
    args = parser.parse_args()
    compiler = args.compiler_build.resolve()
    sys.path.insert(0, str(compiler / "python_packages"))
    from buddy.compiler.frontend import DynamoCompiler
    from buddy.compiler.graph import GraphDriver
    from buddy.compiler.graph.transform import simply_fuse
    from buddy.compiler.ops import tosa
    from buddy.compiler.trace import TraceConfig, load_trace_config

    torch.set_num_threads(4)
    model = load_model(args.weights)
    data = load_input(Path(__file__).parent / "images/dog.bmp")
    with torch.no_grad():
        reference = model(data).numpy()
    design = import_module(args.design)
    quant = import_module(args.design + ".quant.quantize")
    calibration = quant.prepare(model, data)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trace = TraceConfig(load_trace_config(Path(design.__file__).parent / "trace/trace.toml")) if args.trace else None
    decompositions = dict(inductor_decomp)
    remove_decompositions(decompositions, [torch.ops.aten.max_pool2d_with_indices.default])
    importer = DynamoCompiler(primary_registry=tosa.ops_registry,
                              aot_autograd_decomposition=decompositions, trace=trace)
    with torch.no_grad():
        graphs = importer.importer(model, data)
    if len(graphs) != 1:
        raise ValueError(f"AlexNet must import as one graph, got {len(graphs)}")
    graph = graphs[0]
    params = importer.imported_params[graph]
    parameter_names = {id(value): name for name, value in model.named_parameters()}
    names = [parameter_names[id(value)] for value in params]
    graph.fuse_ops([simply_fuse])
    (output / "graph.json").write_text(json.dumps([
        {"name": node.name, "op": type(node).__name__, "args": node.args,
         "shape": node.tensor_meta.get("shape")}
        for node in graph.body
    ], default=str, indent=2))
    quant.quantize(graph, params, names,
                   output, "alexnet", calibration, rax_pack=compiler / "bin/rax-pack")
    driver = GraphDriver(graph)
    for index, subgraph in enumerate(driver.subgraphs):
        subgraph.lower_to_top_level_ir()
        # Chip layouts are already contiguous in the packed payload. Retain
        # their strides while allowing nonzero offsets into that payload.
        from buddy_mlir import ir
        module = subgraph._imported_module
        with module.context:
            function = next(op for op in module.body.operations
                            if op.operation.name == "func.func")
            attributes = []
            for argument in function.regions[0].blocks[0].arguments:
                shape = list(ir.RankedTensorType(argument.type).shape)
                strides = [prod(shape[i + 1:]) for i in range(len(shape))]
                attributes.append(ir.DictAttr.get({
                    "bufferization.buffer_layout": ir.Attribute.parse(
                        f"strided<{strides}, offset: ?>")}))
            function.attributes["arg_attrs"] = ir.ArrayAttr.get(attributes)
        (output / f"subgraph{index}.mlir").write_text(str(subgraph._imported_module))
    (output / "forward.mlir").write_text(str(driver.construct_main_graph(True)))
    data.numpy().tofile(output / "input.f32")
    reference.tofile(output / "reference.f32")
    metadata = {"version": 1, "chip": args.chip, "model": "alexnet",
                "checkpoint": "AlexNet_Weights." + args.weights, "input_shape": [1, 3, 224, 224]}
    (output / "model.json").write_text(json.dumps(metadata))


if __name__ == "__main__":
    main()
