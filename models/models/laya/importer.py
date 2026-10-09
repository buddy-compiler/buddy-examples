from importlib import import_module
import argparse
import importlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from torch._inductor.decomposition import decompositions

from .inputs import encode
from .model import Model, stages

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--compiler-build", type=Path, required=True)
parser.add_argument("--design", required=True)
import_module(".configs.importer-param", __package__).add_arguments(parser)
args = parser.parse_args()
sys.path.insert(0, str(args.compiler_build.resolve() / "python_packages"))
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from buddy_mlir import ir
from stack.compiler.quant.rax import QuantTensor, RaxQuantPackage, write_rax

design = importlib.import_module(args.design + ".design")
quant = importlib.import_module(args.design + ".quant.quantize")
directory = args.output_dir.resolve()
directory.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
checkpoint = snapshot_download(
    args.checkpoint,
    revision=args.revision,
    allow_patterns=[
        "rl_agent_config.json",
        "model.safetensors",
        "encoder/config.json",
        "tokenizer/*",
    ],
)
model = Model(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(Path(checkpoint) / "tokenizer")
tokenizer.save_pretrained(directory / "tokenizer")
config = model.encoder.config
if (
    not 1 <= args.sequence_length <= model.settings["max_len"]
    or not 2 <= args.options <= args.sequence_length
):
    raise ValueError("invalid Laya input capacity")
requests = json.loads(
    (Path(design.__file__).parent / "trace/requests.json").read_text()
)
encoded_samples = [
    encode(tokenizer, request, args.sequence_length, args.options)[0]
    for request in requests
]
input_samples = [
    {key: torch.from_numpy(value) for key, value in encoded.items()}
    for encoded in encoded_samples
]
metadata = {
    "model": args.checkpoint,
    "revision": args.revision,
    "chip": design.chip,
    "linear_format": quant.FORMAT,
    "calibration_samples": len(requests),
    "sequence_length": args.sequence_length,
    "options": args.options,
    "hidden_size": config.hidden_size,
    "actions": len(model.settings["act_costs"]) + 1,
    "settings": model.settings,
    "stages": [],
}
reference = directory / "reference"
reference.mkdir(exist_ok=True)
header = [
    f"constexpr size_t length = {args.sequence_length};",
    f"constexpr size_t width = {config.hidden_size};",
    f"constexpr size_t options = {args.options};",
    f'constexpr size_t actions = {metadata["actions"]};',
]
entries = []
hidden_states = [None] * len(requests)
score_states = [None] * len(requests)
for name, kind, stage in stages(model, args.sequence_length):
    stage.eval()
    samples, results = [], []
    for index, inputs in enumerate(input_samples):
        hidden, scores = hidden_states[index], score_states[index]
        if kind == "embedding":
            sample = {"input_ids": inputs["tokens"]}
        elif kind == "attention":
            sample = {"hidden": hidden, "mask": inputs["mask"]}
        elif kind == "ffn":
            sample = {"hidden": hidden}
        elif kind == "typed":
            sample = {"hidden": hidden, "qtype": inputs["qtype"]}
        elif kind == "scorer":
            markers = hidden.gather(
                1, inputs["positions"][:, :, None].expand(-1, -1, config.hidden_size)
            )
            sample = {"input": markers}
        else:
            values = scores.squeeze(-1).masked_fill(~inputs["valid"].bool(), -1e4)
            p = values.softmax(-1)
            count = inputs["valid"].sum(-1).clamp(min=2).float()
            entropy = -(p * p.clamp_min(1e-9).log()).sum(-1) / count.log()
            top = p.topk(2, -1).values
            features = torch.stack(
                (top[:, 0], top[:, 0] - top[:, 1], entropy, count / 255), -1
            )
            sample = {"input": torch.cat((hidden[:, 0], features), -1)}
        with torch.no_grad():
            result = stage(**sample)
        samples.append(sample)
        results.append(result)
    sample, result = samples[0], results[0]
    result.detach().numpy().tofile(reference / f"{name}.f32")
    if kind in quant.STAGES and quant.PREPARE is not None:
        quant.PREPARE(stage, samples)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=decompositions,
        func_name=f"forward_{name}",
    )
    with torch.no_grad():
        graphs = compiler.importer(stage, **sample)
    if len(graphs) != 1:
        raise ValueError(f"{name}: expected one graph")
    graph = graphs[0]
    params = compiler.imported_params[graph]
    names = {
        value.data_ptr(): key
        for key, value in list(stage.named_parameters()) + list(stage.named_buffers())
    }
    order = sorted(
        zip(graph._fake_params, params), key=lambda item: names[item[1].data_ptr()]
    )
    graph._fake_params = [index for index, _ in order]
    params[:] = [value for _, value in order]
    input_order = {
        value.data_ptr(): index
        for value, index in zip(graph._runtime_inputs_ref, graph._inputs)
    }
    graph._inputs = [input_order[value.data_ptr()] for value in sample.values()]
    graph._runtime_inputs_ref = list(sample.values())
    graph.fuse_ops([simply_fuse])
    output = directory / name
    output.mkdir(exist_ok=True)
    parameter_names = [names[value.data_ptr()] for value in params]
    if kind in quant.STAGES:
        graph.packing_target = design.targets[kind]
        quant.apply(
            graph,
            params,
            parameter_names,
            output,
            name,
            args.compiler_build / "bin/rax-pack",
            stage,
            samples,
        )
    else:
        tensors, data, offset = [], [], 0
        for key, value in zip(parameter_names, params):
            raw = value.detach().numpy().tobytes()
            tensors.append(
                QuantTensor(
                    key,
                    list(value.shape),
                    list(value.shape),
                    "f32",
                    [],
                    offset,
                    len(raw),
                    0,
                    0,
                )
            )
            data.append(raw)
            offset += len(raw)
        payload = RaxQuantPackage(tensors, b"", b"".join(data), b"", {})
        write_rax(
            payload, output / f"{name}.rax", args.compiler_build / "bin/rax-pack", name
        )
        (output / "params.f32").write_bytes(payload.params_f32)
        (output / "weights.bin").write_bytes(b"")
    symbol = f"subgraph_{name}"
    graph.op_groups[symbol] = graph.op_groups.pop("subgraph0")
    graph.group_map_device[symbol] = graph.device
    driver = GraphDriver(graph)
    subgraph = driver.subgraphs[0]
    subgraph.lower_to_top_level_ir()
    module = subgraph._imported_module
    with module.context:
        function = next(
            op for op in module.body.operations if op.operation.name == "func.func"
        )
        attrs = []
        for argument in function.regions[0].blocks[0].arguments:
            shape = list(ir.RankedTensorType(argument.type).shape)
            strides = [math.prod(shape[i + 1 :]) for i in range(len(shape))]
            attrs.append(
                ir.DictAttr.get(
                    {
                        "bufferization.buffer_layout": ir.Attribute.parse(
                            f"strided<{strides}, offset: ?>"
                        ),
                        "bufferization.writable": ir.BoolAttr.get(False),
                    }
                )
            )
        function.attributes["arg_attrs"] = ir.ArrayAttr.get(attrs)
    (directory / f"{name}.mlir").write_text(str(module))
    forward = driver.construct_main_graph(True)
    target = design.targets[kind]
    with forward.context:
        for op in forward.body.operations:
            if (
                op.operation.name == "func.func"
                and ir.StringAttr(op.attributes["sym_name"]).value == symbol
            ):
                op.attributes["buckyball.target"] = ir.StringAttr.get(target)
    (directory / f"{name}-forward.mlir").write_text(str(forward))
    quantized = any(value.dtype == torch.int8 for value in params)
    types = ["Matrix *" if kind == "action" else "Hidden *", "Floats *"]
    arguments = [
        (
            "&ctx.action"
            if kind == "action"
            else "&ctx.scores" if kind == "scorer" else "&ctx.next"
        ),
        "&params.floats",
    ]
    if quantized:
        types.append("Bytes *")
        arguments.append("&params.bytes")
    abi = {
        "embedding": (["Tokens *"], ["&ctx.tokens"]),
        "attention": (["Hidden *", "Tokens *"], ["&ctx.hidden", "&ctx.mask"]),
        "ffn": (["Hidden *"], ["&ctx.hidden"]),
        "typed": (["Hidden *", "Index *"], ["&ctx.hidden", "&ctx.qtype"]),
        "scorer": (["Hidden *"], ["&ctx.markers"]),
        "action": (["Matrix *"], ["&ctx.features"]),
    }
    inputs_types, inputs_args = abi[kind]
    types += inputs_types
    arguments += inputs_args
    header.append(f'extern "C" void _mlir_ciface_forward_{name}({", ".join(types)});')
    header.append(
        f'static void run_{name}(Context &ctx, Parameters &params) {{ _mlir_ciface_forward_{name}({", ".join(arguments)}); }}'
    )
    entries.append(f'{{"{name}", "{kind}", run_{name}}}')
    metadata["stages"].append(
        {"name": name, "kind": kind, "target": target, "quantized": quantized}
    )
    if kind == "scorer":
        score_states = results
    elif kind != "action":
        hidden_states = results
    print(f"exported {name} -> {target}", flush=True)
header.append("const Entry entries[] = {" + ",\n".join(entries) + "};")
(directory / "stages.h").write_text("\n".join(header) + "\n")
(directory / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
for key, value in encoded_samples[0].items():
    value.tofile(reference / f"{key}.i64")
