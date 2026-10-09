import argparse
from importlib import import_module
import json
import hashlib
from pathlib import Path
import sys

import torch
from torch._inductor.decomposition import decompositions as inductor_decomp

from .prepare import load_input, load_model


def main():
    parser = argparse.ArgumentParser(description="MLP-Mixer AOT importer")
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
    from buddy.compiler.graph.transform import simply_fuse
    from buddy.compiler.ops import tosa
    from buddy.compiler.trace import TraceConfig, load_trace_config

    torch.set_num_threads(4)
    model = load_model(args.checkpoint, args.revision)
    data = load_input(Path(__file__).parent.parent / "alexnet/images/dog.bmp")
    with torch.no_grad():
        reference = model(data).numpy()
    design = import_module(args.design)
    quant = import_module(args.design + ".quant.quantize")
    calibration = quant.prepare(model, data)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trace = TraceConfig(load_trace_config(Path(design.__file__).parent / "trace/trace.toml")) if args.trace else None
    decompositions = dict(inductor_decomp)
    importer = DynamoCompiler(primary_registry=tosa.ops_registry,
                              aot_autograd_decomposition=decompositions, trace=trace)
    with torch.no_grad():
        graphs = importer.importer(model, data)
    if len(graphs) != 1:
        raise ValueError(f"MLP-Mixer must import as one graph, got {len(graphs)}")
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
                   output, "mixer", calibration, rax_pack=compiler / "bin/rax-pack")
    (output / "quant-graph.json").write_text(json.dumps([
        {"name": node.name, "op": type(node).__name__, "args": node.args,
         "parents": node._parents, "shape": node.tensor_meta.get("shape")}
        for node in graph.body
    ], default=str, indent=2))
    shapes = import_module(args.design + ".lowering").emit(graph, output)
    offsets, offset = {}, 0
    for name, parameter in zip(names, params):
        if parameter.dtype == torch.float32:
            offsets[name] = offset
            offset += parameter.numel()
    norm_names = [f"blocks.{block}.norm{norm}" for block in range(12) for norm in (1, 2)] + ["norm"]
    header = ["#pragma once", "using Linear = void (*)(StridedMemRefType<float, 2> *, MemRef<float, 1> *, MemRef<int8_t, 1> *, StridedMemRefType<float, 2> *);"]
    header += [f'extern "C" void _mlir_ciface_linear{i}(StridedMemRefType<float, 2> *, MemRef<float, 1> *, MemRef<int8_t, 1> *, StridedMemRefType<float, 2> *);' for i in range(50)]
    header += ["static Linear linears[] = {" + ",".join(f"_mlir_ciface_linear{i}" for i in range(50)) + "};"]
    header += ["static const size_t shapes[50][4] = {" + ",".join("{" + ",".join(map(str, shape)) + "}" for shape in shapes) + "};"]
    header += ["static const size_t norms[25][2] = {" + ",".join("{" + str(offsets[name + ".weight"]) + "," + str(offsets[name + ".bias"]) + "}" for name in norm_names) + "};"]
    (output / "mixer-parameters.h").write_text("\n".join(header))
    data.numpy().tofile(output / "input.f32")
    reference.tofile(output / "reference.f32")
    metadata = {"version": 1, "chip": args.chip, "model": "mlp-mixer",
                "checkpoint": args.checkpoint, "revision": args.revision, "input_shape": [196, 768],
                "input_sha256": hashlib.sha256(data.numpy().tobytes()).hexdigest(),
                "compute_cores": 4, "weight_dtype": "symmetric_int8_per_output_channel",
                "accumulator_dtype": "int32"}
    (output / "model.json").write_text(json.dumps(metadata))


if __name__ == "__main__":
    main()
