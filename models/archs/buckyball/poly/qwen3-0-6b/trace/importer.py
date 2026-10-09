from importlib import import_module

import filecmp
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from torch._inductor.decomposition import decompositions

from .stages import Attention, Embedding, FFN


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--compiler-build", type=Path, required=True)
import_module("..configs.importer-param", __package__).add_arguments(parser)
args = parser.parse_args()
args.compiler_build = args.compiler_build.resolve()
if not 1 <= args.prefill_len <= args.max_cache_len:
    raise ValueError("prefill length must be within the cache capacity")
sys.path.insert(0, str(args.compiler_build.resolve() / "python_packages"))
design = import_module(__package__.removesuffix(".trace"))
quant = import_module(design.__name__ + ".quant.quantize")
layout = import_module(design.__name__ + ".permute.layout")
design_dir = Path(design.__file__).parent

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from buddy_mlir import ir

from .ffn import FFNExpand, FFNDown, ffn_widths, parameters as ffn_parameters
from .attention import (
    AttentionBody,
    AttentionProjection,
    parameters as attention_parameters,
)
from ..quant.parameters import write_parameters
from examples.balls.mxmm.compiler.python.layout import bank_bytes
from .output import OutputShard

torch.set_num_threads(4)
directory = args.output_dir.resolve()
directory.mkdir(parents=True, exist_ok=True)
model_id = args.model
model, tokenizer = import_module("stack.models.models.qwen3-0-6b.prepare").prepare(
    model_id
)
model.set_attn_implementation("eager")
config = model.config
if config.model_type != "qwen3" or config.attention_bias:
    raise ValueError("the compiled pipeline requires bias-free Qwen3 attention")
head_dim = config.head_dim
if args.parts < 1 or any(
    size % args.parts
    for size in (
        config.num_attention_heads,
        config.num_key_value_heads,
        config.intermediate_size,
    )
):
    raise ValueError(
        "tile count must divide Q heads, KV heads and FFN intermediate channels"
    )
if any(
    size % (2 * args.parts)
    for size in (config.num_attention_heads, config.num_key_value_heads)
):
    raise ValueError("paired execution tiles must divide Q and KV heads")
if config.use_sliding_window:
    raise ValueError("this artifact requires full causal attention")
tokenizer.save_pretrained(directory / "tokenizer")
config.save_pretrained(directory / "tokenizer")
model.generation_config.save_pretrained(directory / "tokenizer")
tokenizer_config_path = directory / "tokenizer/tokenizer_config.json"
tokenizer_config = json.loads(tokenizer_config_path.read_text())
tokenizer_config["tokenizer_class"] = "PreTrainedTokenizerFast"
tokenizer_config_path.write_text(json.dumps(tokenizer_config, indent=2))
tracing = import_module(design.__name__ + ".trace.inputs")
prompt = tracing.sample_tokens(tokenizer)
if prompt.shape[1] >= args.max_cache_len:
    raise ValueError("example prompt leaves no room for a decode token")
inputs = torch.zeros((1, args.prefill_len), dtype=torch.int64)
if prompt.shape[1] > args.prefill_len:
    raise ValueError("example prompt exceeds the compiled prefill length")
inputs[:, : prompt.shape[1]] = prompt
mask = torch.zeros_like(inputs)
mask[:, : prompt.shape[1]] = 1
captured = {}
hooks = []
for index, layer in enumerate(model.model.layers):
    hooks.append(
        layer.register_forward_pre_hook(
            lambda module, values, index=index: captured.update(
                {("attention", index): values[0].detach().clone()}
            )
        )
    )
    hooks.append(
        layer.post_attention_layernorm.register_forward_pre_hook(
            lambda module, values, index=index: captured.update(
                {("ffn", index): values[0].detach().clone()}
            )
        )
    )
hooks.append(
    model.model.norm.register_forward_pre_hook(
        lambda module, values: captured.update(
            {("output", 0): values[0].detach().clone()}
        )
    )
)
with torch.no_grad():
    model(input_ids=inputs, attention_mask=mask, use_cache=False, logits_to_keep=1)
prefill_samples = dict(captured)
captured.clear()
with torch.no_grad():
    reference = model(input_ids=prompt, use_cache=True, logits_to_keep=1)
    next_token = reference.logits[:, -1].argmax(-1, keepdim=True)
    cache = reference.past_key_values
    model(input_ids=next_token, past_key_values=cache, use_cache=True, logits_to_keep=1)
decode_samples = dict(captured)
for hook in hooks:
    hook.remove()

metadata = {
    "chip": design.chip,
    "version": 3,
    "weight_format": "mxfp8_e4m3",
    "parts": args.parts,
    "num_attention_heads": config.num_attention_heads,
    "model": model_id,
    "model_type": config.model_type,
    "hidden_size": config.hidden_size,
    "num_layers": config.num_hidden_layers,
    "num_kv_heads": config.num_key_value_heads,
    "head_dim": head_dim,
    "vocab_size": config.vocab_size,
    "prefill_length": args.prefill_len,
    "kv_cache_format": "mxfp8_e4m3",
    "kv_cache_block_size": 32,
    "kv_cache_scale_format": "e8m0",
    "kv_cache_scale_selection": "cover_maximum_f32_finite",
    "kv_cache_storage": "separate_codes_scales",
    "cache_length": args.max_cache_len,
    "workspace_bytes": 128 * 1024 * 1024,
    "parameter_sha256": {
        name: hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()
        for name, value in model.named_parameters()
    },
    "parameter_aliases": (
        {"lm_head.weight": "model.embed_tokens.weight"}
        if model.lm_head.weight is model.model.embed_tokens.weight
        else {}
    ),
    "stages": [],
}


def export(name, kind, stage, sample, shared=None, *, record=True):
    stage.eval()
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=decompositions,
        func_name=f"forward_{name}",
    )
    quant.cache.register(compiler)
    with torch.no_grad():
        graphs = compiler.importer(stage, **sample)
    if len(graphs) != 1:
        raise ValueError(f"{name}: expected one captured subgraph")
    graph = graphs[0]
    params = compiler.imported_params[graph]
    names = {
        value.data_ptr(): key
        for key, value in list(stage.named_parameters()) + list(stage.named_buffers())
    }
    ordered = sorted(
        zip(graph._fake_params, params), key=lambda item: names[item[1].data_ptr()]
    )
    graph._fake_params = [index for index, _ in ordered]
    params[:] = [value for _, value in ordered]
    order = {
        value.data_ptr(): index
        for value, index in zip(graph._runtime_inputs_ref, graph._inputs)
    }
    graph._inputs = [order[value.data_ptr()] for value in sample.values()]
    graph._runtime_inputs_ref = list(sample.values())
    graph.fuse_ops([simply_fuse])
    graph.packing_rows = args.prefill_len
    layout.apply(graph, kind=kind)
    output = directory / name
    quant.quantize(
        graph,
        params,
        [names[value.data_ptr()] for value in params],
        output,
        name,
        args.compiler_build / "bin/rax-pack",
        kind=kind,
    )
    symbol = f"subgraph_{name}"
    graph.op_groups[symbol] = graph.op_groups.pop("subgraph0")
    graph.group_map_device[symbol] = graph.device
    driver = GraphDriver(graph)
    subgraph = driver.subgraphs[0]
    subgraph.lower_to_top_level_ir()
    target = design.targets[kind]
    module = subgraph._imported_module
    with module.context:
        function = next(
            op
            for op in module.body.operations
            if op.operation.name == "func.func"
            and ir.StringAttr(op.attributes["sym_name"]).value == symbol
        )
        argument_attrs = []
        for argument in function.regions[0].blocks[0].arguments:
            shape = list(ir.RankedTensorType(argument.type).shape)
            strides = [math.prod(shape[index + 1 :]) for index in range(len(shape))]
            argument_attrs.append(
                ir.DictAttr.get(
                    {
                        "bufferization.buffer_layout": ir.Attribute.parse(
                            f"strided<{strides}, offset: ?>"
                        ),
                        "bufferization.writable": ir.BoolAttr.get(False),
                    }
                )
            )
        function.attributes["arg_attrs"] = ir.ArrayAttr.get(argument_attrs)
    (directory / f"{name}.mlir").write_text(str(module))
    forward = driver.construct_main_graph(True)
    with forward.context:
        for op in forward.body.operations:
            if (
                op.operation.name == "func.func"
                and ir.StringAttr(op.attributes["sym_name"]).value == symbol
            ):
                op.attributes["buckyball.target"] = ir.StringAttr.get(target)
    (directory / f"{name}-forward.mlir").write_text(str(forward))
    parameter_dir = name
    if shared is not None:
        for filename in ("params.f32", "weights.bin"):
            if not filecmp.cmp(
                output / filename, directory / shared / filename, shallow=False
            ):
                raise ValueError(f"{name}: phase-dependent packed parameter {filename}")
        parameter_dir = shared
    entry = {
        "name": name,
        "kind": kind,
        "target": target,
        "parameters": parameter_dir,
        "f32_elements": sum(
            value.numel() for value in params if value.dtype == torch.float32
        ),
        "weight_bytes": sum(
            value.numel() for value in params if value.dtype == torch.int8
        ),
        "bank_bytes": (
            0 if kind == "embedding" else bank_bytes(args.compiler_build, target)
        ),
    }
    if kind == "embedding":
        entry["weight_alignment"] = 16
    if record:
        metadata["stages"].append(entry)
    print(f"exported {name} -> {target}", flush=True)


local_intermediate = config.intermediate_size // args.parts
expand_widths = ffn_widths(local_intermediate // 2, 32)
down_widths = ffn_widths(config.hidden_size // 2, 16)
local_query = config.num_attention_heads // args.parts * head_dim
metadata["ffn"] = {
    "workers": 3,
    "intermediate": local_intermediate,
    "expand_rows": local_intermediate // 2,
    "down_rows": config.hidden_size // 2,
    "expand_widths": expand_widths,
    "down_widths": down_widths,
}
metadata["attention"] = {
    "kv_heads": config.num_key_value_heads // args.parts // 2,
    "context": local_query // 2,
    "projection_input": local_query,
}
metadata["execution_tiles"] = 2 * args.parts
query_heads = config.num_attention_heads // metadata["execution_tiles"]
kv_heads = config.num_key_value_heads // metadata["execution_tiles"]
metadata["attention"]["shards"] = []
for rank in range(metadata["execution_tiles"]):
    head_group = rank % args.parts * 2 + rank // args.parts
    metadata["attention"]["shards"].append(
        {
            "rank": rank,
            "query_heads": list(
                range(head_group * query_heads, (head_group + 1) * query_heads)
            ),
            "kv_heads": list(range(head_group * kv_heads, (head_group + 1) * kv_heads)),
            "kv_owner_rank": rank,
            "kv_consumer_ranks": [rank],
            "projection_peer_rank": (rank + args.parts) % metadata["execution_tiles"],
        }
    )

for phase, samples, length in (
    ("prefill", prefill_samples, args.prefill_len),
    ("decode", decode_samples, 1),
):
    attention_positions = (
        torch.arange(length) if phase == "prefill" else torch.tensor([prompt.shape[1]])
    )
    attention_cache_shape = (
        1,
        config.num_key_value_heads // args.parts // 2,
        args.max_cache_len,
        head_dim,
    )
    attention_scale_shape = (*attention_cache_shape[:-1], head_dim // 32)
    export(
        f"{phase}_attention_body",
        "attention",
        AttentionBody(
            model.model.layers[0],
            config,
            args.max_cache_len,
            model.model.rotary_emb.inv_freq,
            0,
            args.parts,
            model.model.rotary_emb.attention_scaling,
            quant.cache,
        ),
        {
            "hidden": samples["attention", 0],
            "key_codes": torch.zeros(attention_cache_shape, dtype=torch.int8),
            "key_scales": torch.full(attention_scale_shape, 127, dtype=torch.int8),
            "value_codes": torch.zeros(attention_cache_shape, dtype=torch.int8),
            "value_scales": torch.full(attention_scale_shape, 127, dtype=torch.int8),
            "positions": attention_positions,
        },
        record=False,
    )
    generic_attention = Attention(
        model.model.layers[0],
        config,
        args.max_cache_len,
        model.model.rotary_emb.inv_freq,
        0,
        args.parts,
        model.model.rotary_emb.attention_scaling,
        quant.cache,
    )
    export(
        f"{phase}_attention_projection",
        "attention",
        AttentionProjection(generic_attention),
        {"hidden": torch.zeros((1, length, local_query))},
        record=False,
    )
    generic_ffn = FFN(model.model.layers[0], 0, args.parts)
    for width in sorted(set(expand_widths)):
        export(
            f"{phase}_ffn_expand_{width}",
            "ffn",
            FFNExpand(generic_ffn, width),
            {"hidden": samples["ffn", 0]},
            record=False,
        )
    for width in sorted(set(down_widths)):
        export(
            f"{phase}_ffn_down_{width}",
            "ffn",
            FFNDown(generic_ffn, width),
            {"hidden": torch.zeros((1, length, local_intermediate))},
            record=False,
        )
    token_ids = inputs if phase == "prefill" else next_token
    export(
        f"{phase}_embedding",
        "embedding",
        Embedding(model.model.embed_tokens),
        {"input_ids": token_ids},
        "prefill_embedding" if phase == "decode" else None,
    )
    for rank in range(metadata["execution_tiles"]):
        for index, layer in enumerate(model.model.layers):
            for kind, values in (
                (
                    "attention",
                    attention_parameters(
                        layer, config, model.model.rotary_emb.inv_freq, rank, args.parts
                    ),
                ),
                (
                    "ffn",
                    ffn_parameters(layer, rank, args.parts, expand_widths, down_widths),
                ),
            ):
                name = f"{phase}_{kind}_{index}_rank_{rank}"
                parameter_dir = f"prefill_{kind}_{index}_rank_{rank}"
                if phase == "prefill":
                    specification = write_parameters(
                        values,
                        directory / parameter_dir,
                        parameter_dir,
                        args.compiler_build / "bin/rax-pack",
                        design.targets[kind],
                        args.prefill_len,
                    )
                    if rank == 0 and index == 0:
                        metadata[kind]["matrices"] = specification["matrices"]
                        metadata[kind]["bank_bytes"] = specification["bank_bytes"]
                else:
                    original = next(
                        entry
                        for entry in metadata["stages"]
                        if entry["name"] == parameter_dir
                    )
                    specification = {
                        key: original[key]
                        for key in ("f32_elements", "weight_bytes", "bank_bytes")
                    }
                metadata["stages"].append(
                    {
                        "name": name,
                        "kind": kind,
                        "target": design.targets[kind],
                        "parameters": parameter_dir,
                        **{
                            key: specification[key]
                            for key in ("f32_elements", "weight_bytes", "bank_bytes")
                        },
                    }
                )
export(
    "output",
    "output",
    OutputShard(model, 0, metadata["execution_tiles"]),
    {"hidden": decode_samples["output", 0]},
    record=False,
)
for rank in range(metadata["execution_tiles"]):
    name = f"output_rank_{rank}"
    stage = OutputShard(model, rank, metadata["execution_tiles"])
    specification = write_parameters(
        dict(stage.named_parameters()),
        directory / name,
        name,
        args.compiler_build / "bin/rax-pack",
        design.targets["output"],
        args.prefill_len,
    )
    metadata["stages"].append(
        {
            "name": name,
            "kind": "output",
            "target": design.targets["output"],
            "parameters": name,
            **{
                key: specification[key]
                for key in ("f32_elements", "weight_bytes", "bank_bytes")
            },
        }
    )
(directory / "model.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
header = [
    f"constexpr size_t hiddenSize = {config.hidden_size};",
    f"constexpr size_t layers = {config.num_hidden_layers};",
    f"constexpr size_t kvHeads = {metadata['attention']['kv_heads']};",
    f"constexpr size_t attentionContext = {metadata['attention']['context']};",
    f"constexpr size_t attentionProjectionInput = {local_query};",
    f"constexpr size_t parts = {args.parts};",
    f"constexpr size_t executionTiles = {metadata['execution_tiles']};",
    f"constexpr size_t ffnIntermediate = {local_intermediate};",
    f"constexpr size_t headSize = {head_dim};",
    f"constexpr size_t vocabulary = {config.vocab_size};",
    f"constexpr size_t prefillLength = {args.prefill_len};",
    f"constexpr size_t cacheLength = {args.max_cache_len};",
    f"constexpr size_t workspaceBytes = {metadata['workspace_bytes']};",
]
declarations = {
    "embedding": "Hidden *, Bytes *, Tokens *",
    "output": "Hidden *, Floats *, Bytes *, Hidden *",
}
for phase in ("prefill", "decode"):
    header.append(
        f'extern "C" void _mlir_ciface_subgraph_{phase}_attention_body('
        "AttentionBodyResult *, Hidden *, AttentionFloatView *, AttentionWeightView *, "
        "AttentionFloatView *, AttentionWeightView *, AttentionFloatView *, AttentionWeightView *, "
        "Positions *, AttentionFloatView *, Cache *, Cache *, Cache *, Cache *);"
    )
    header.append(
        f'extern "C" void _mlir_ciface_subgraph_{phase}_attention_projection('
        "Hidden *, Hidden *, AttentionWeightView *);"
    )

    def matrix_range(value):
        return "{" + f"{value['offset']}, {value['bytes']}" + "}"

    matrices = metadata["attention"]["matrices"]
    header.append(
        f"const AttentionKernels {phase}AttentionKernels = "
        f"{{{head_dim}, "
        + ", ".join(
            matrix_range(matrices[key + ".weight"])
            for key in ("q_proj", "k_proj", "v_proj", "o_proj")
        )
        + ", "
        f"_mlir_ciface_subgraph_{phase}_attention_body, _mlir_ciface_subgraph_{phase}_attention_projection}};"
    )
    for kind, widths, signature in (
        (
            "expand",
            expand_widths,
            "Hidden *, Hidden *, Floats *, FfnWeightView *, FfnWeightView *",
        ),
        ("down", down_widths, "Hidden *, Hidden *, FfnWeightView *"),
    ):
        for width in sorted(set(widths)):
            header.append(
                f'extern "C" void _mlir_ciface_subgraph_{phase}_ffn_{kind}_{width}({signature});'
            )

    def sizes(values):
        return "{" + ", ".join(map(str, values)) + "}"

    kernels = (
        lambda kind, values: "{"
        + ", ".join(
            f"_mlir_ciface_subgraph_{phase}_ffn_{kind}_{width}" for width in values
        )
        + "}"
    )
    header.append(
        f"const FfnKernels {phase}FfnKernels = {{"
        + ", ".join(
            (
                str(config.hidden_size),
                str(local_intermediate),
                *[
                    "{{"
                    + ", ".join(
                        matrix_range(
                            metadata["ffn"]["matrices"][f"{kind}_{index}.weight"]
                        )
                        for index in range(3)
                    )
                    + "}}"
                    for kind in ("gate", "up", "down")
                ],
                sizes(expand_widths),
                sizes(down_widths),
                kernels("expand", expand_widths),
                kernels("down", down_widths),
            )
        )
        + "};"
    )
for entry in metadata["stages"]:
    name, kind = entry["name"], entry["kind"]
    if kind == "attention":
        runner = (
            "&prefillAttentionKernels"
            if name.startswith("prefill_")
            else "&decodeAttentionKernels"
        )
    elif kind == "ffn":
        runner = (
            "&prefillFfnKernels" if name.startswith("prefill_") else "&decodeFfnKernels"
        )
    else:
        runner = (
            "_mlir_ciface_forward_output"
            if kind == "output"
            else f"_mlir_ciface_forward_{name}"
        )
        header.append(f'extern "C" void {runner}({declarations[kind]});')
    alignment = (
        entry["weight_alignment"] if kind == "embedding" else entry["bank_bytes"]
    )
    parameter = '{"%s", %d, %d, %d}' % (
        entry["parameters"],
        entry["f32_elements"],
        entry["weight_bytes"],
        alignment,
    )
    header.append(f"const {kind.capitalize()}Entry {name} = {{{parameter}, {runner}}};")
header.append(
    "const OutputEntry output[executionTiles] = {"
    + ", ".join(f"output_rank_{rank}" for rank in range(metadata["execution_tiles"]))
    + "};"
)
for phase in ("prefill", "decode"):
    for kind in ("attention", "ffn"):
        rows = [
            "{"
            + ", ".join(
                f"{phase}_{kind}_{index}_rank_{rank}"
                for index in range(config.num_hidden_layers)
            )
            + "}"
            for rank in range(metadata["execution_tiles"])
        ]
        header.append(
            f"const {kind.capitalize()}Entry {phase}_{kind}[executionTiles][layers] = {{"
            + ", ".join(rows)
            + "};"
        )
(directory / "qwen-parameters.h").write_text("\n".join(header) + "\n")
