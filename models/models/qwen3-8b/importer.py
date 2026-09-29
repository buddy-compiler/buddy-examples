from importlib import import_module

import filecmp
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from torch._inductor.decomposition import decompositions
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from .stages import Attention, Embedding, FFN, Output


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--compiler-build", type=Path, required=True)
parser.add_argument("--design", required=True)
import_module(".configs.importer-param", __package__).add_arguments(parser)
args = parser.parse_args()
args.compiler_build = args.compiler_build.resolve()
if not 1 <= args.prefill_len <= args.max_cache_len:
    raise ValueError("prefill length must be within the cache capacity")
sys.path.insert(0, str(args.compiler_build.resolve() / "python_packages"))
design = import_module(args.design)
quant = import_module(design.__name__ + ".quant.quantize")
design_dir = Path(design.__file__).parent

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from buddy_mlir import ir


torch.set_num_threads(4)
directory = args.output_dir.resolve()
directory.mkdir(parents=True, exist_ok=True)
model_id = args.model
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.float32, attn_implementation="eager"
).eval()
config = model.config
if config.model_type != "qwen3" or config.attention_bias:
    raise ValueError("the compiled pipeline requires bias-free Qwen3 attention")
head_dim = config.head_dim
if args.parts < 1 or any(size % args.parts for size in
                         (config.num_attention_heads, config.num_key_value_heads, config.intermediate_size)):
    raise ValueError("tile count must divide Q heads, KV heads and FFN intermediate channels")
if config.use_sliding_window:
    raise ValueError("this artifact requires full causal attention")
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
tokenizer.save_pretrained(directory / "tokenizer")
config.save_pretrained(directory / "tokenizer")
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
inputs[:, :prompt.shape[1]] = prompt
mask = torch.zeros_like(inputs)
mask[:, :prompt.shape[1]] = 1
captured = {}
hooks = []
for index, layer in enumerate(model.model.layers):
    hooks.append(layer.register_forward_pre_hook(
        lambda module, values, index=index: captured.update({("attention", index): values[0].detach().clone()})
    ))
    hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(
        lambda module, values, index=index: captured.update({("ffn", index): values[0].detach().clone()})
    ))
hooks.append(model.model.norm.register_forward_pre_hook(
    lambda module, values: captured.update({("output", 0): values[0].detach().clone()})
))
with torch.no_grad():
    model(input_ids=inputs, attention_mask=mask, use_cache=False, logits_to_keep=1)
prefill_samples = dict(captured)
captured.clear()
with torch.no_grad():
    reference = model(input_ids=prompt, use_cache=True, logits_to_keep=1)
    next_token = reference.logits[:, -1].argmax(-1, keepdim=True)
    cache = reference.past_key_values
    saved_keys = [layer.keys.detach().clone() for layer in cache.layers]
    saved_values = [layer.values.detach().clone() for layer in cache.layers]
    model(input_ids=next_token, past_key_values=cache, use_cache=True, logits_to_keep=1)
decode_samples = dict(captured)
for hook in hooks:
    hook.remove()

metadata = {
    "chip": design.chip,
    "version": 3, "weight_format": "mxfp8_e4m3", "parts": args.parts, "num_attention_heads": config.num_attention_heads, "model": model_id, "model_type": config.model_type,
    "hidden_size": config.hidden_size, "num_layers": config.num_hidden_layers,
    "num_kv_heads": config.num_key_value_heads, "head_dim": head_dim,
    "vocab_size": config.vocab_size, "prefill_length": args.prefill_len,
    "cache_length": args.max_cache_len, "workspace_bytes": 128 * 1024 * 1024,
    "parameter_sha256": {
        name: hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()
        for name, value in model.named_parameters()
    },
    "parameter_aliases": {"lm_head.weight": "model.embed_tokens.weight"}
                         if model.lm_head.weight is model.model.embed_tokens.weight else {},
    "stages": [],
}


def export(name, kind, stage, sample, shared=None):
    stage.eval()
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=decompositions,
        func_name=f"forward_{name}",
    )
    with torch.no_grad():
        graphs = compiler.importer(stage, **sample)
    if len(graphs) != 1:
        raise ValueError(f"{name}: expected one captured subgraph")
    graph = graphs[0]
    params = compiler.imported_params[graph]
    names = {value.data_ptr(): key for key, value in
             list(stage.named_parameters()) + list(stage.named_buffers())}
    ordered = sorted(zip(graph._fake_params, params), key=lambda item: names[item[1].data_ptr()])
    graph._fake_params = [index for index, _ in ordered]
    params[:] = [value for _, value in ordered]
    order = {value.data_ptr(): index for value, index in zip(graph._runtime_inputs_ref, graph._inputs)}
    graph._inputs = [order[value.data_ptr()] for value in sample.values()]
    graph._runtime_inputs_ref = list(sample.values())
    graph.fuse_ops([simply_fuse])
    output = directory / name
    quant.quantize(graph, params, [names[value.data_ptr()] for value in params],
                    output, name, args.compiler_build / "bin/rax-pack", kind=kind)
    symbol = f"subgraph_{name}"
    graph.op_groups[symbol] = graph.op_groups.pop("subgraph0")
    graph.group_map_device[symbol] = graph.device
    driver = GraphDriver(graph)
    subgraph = driver.subgraphs[0]
    subgraph.lower_to_top_level_ir()
    target = design.targets[kind]
    module = subgraph._imported_module
    with module.context:
        function = next(op for op in module.body.operations if op.operation.name == "func.func")
        argument_attrs = []
        for argument in function.regions[0].blocks[0].arguments:
            shape = list(ir.RankedTensorType(argument.type).shape)
            strides = [math.prod(shape[index+1:]) for index in range(len(shape))]
            argument_attrs.append(ir.DictAttr.get({
                "bufferization.buffer_layout": ir.Attribute.parse(f"strided<{strides}, offset: ?>"),
                "bufferization.writable": ir.BoolAttr.get(False),
            }))
        function.attributes["arg_attrs"] = ir.ArrayAttr.get(argument_attrs)
    (directory / f"{name}.mlir").write_text(str(module))
    forward = driver.construct_main_graph(True)
    with forward.context:
        for op in forward.body.operations:
            if op.operation.name == "func.func" and ir.StringAttr(op.attributes["sym_name"]).value == symbol:
                op.attributes["buckyball.target"] = ir.StringAttr.get(target)
    (directory / f"{name}-forward.mlir").write_text(str(forward))
    parameter_dir = name
    if shared is not None:
        for filename in ("params.f32", "weights.bin"):
            if not filecmp.cmp(output / filename, directory / shared / filename, shallow=False):
                raise ValueError(f"{name}: phase-dependent packed parameter {filename}")
        parameter_dir = shared
    entry = {
        "name": name, "kind": kind, "target": target, "parameters": parameter_dir,
        "f32_elements": sum(value.numel() for value in params if value.dtype == torch.float32),
        "weight_bytes": sum(value.numel() for value in params if value.dtype == torch.int8),
    }
    metadata["stages"].append(entry)
    print(f"exported {name} -> {target}", flush=True)


cache_shape = (1, config.num_key_value_heads // args.parts, args.max_cache_len, head_dim)
for phase, samples, length in (
    ("prefill", prefill_samples, args.prefill_len),
    ("decode", decode_samples, 1),
):
    token_ids = inputs if phase == "prefill" else next_token
    export(f"{phase}_embedding", "embedding", Embedding(model.model.embed_tokens),
           {"input_ids": token_ids}, "prefill_embedding" if phase == "decode" else None)
    for rank in range(args.parts):
        head_begin = rank * cache_shape[1]
        head_end = head_begin + cache_shape[1]
        for index, layer in enumerate(model.model.layers):
            keys = torch.zeros(cache_shape)
            values = torch.zeros(cache_shape)
            if phase == "decode":
                keys[:, :, :prompt.shape[1]] = saved_keys[index][:, head_begin:head_end]
                values[:, :, :prompt.shape[1]] = saved_values[index][:, head_begin:head_end]
            positions = torch.arange(length) if phase == "prefill" else torch.tensor([prompt.shape[1]])
            export(f"{phase}_attention_{index}_rank_{rank}", "attention",
                   Attention(layer, config, args.max_cache_len, model.model.rotary_emb.inv_freq, rank, args.parts,
                             model.model.rotary_emb.attention_scaling),
                   {"hidden": samples["attention", index], "keys": keys,
                    "values": values, "positions": positions},
                   f"prefill_attention_{index}_rank_{rank}" if phase == "decode" else None)
            export(f"{phase}_ffn_{index}_rank_{rank}", "ffn", FFN(layer, rank, args.parts),
                   {"hidden": samples["ffn", index]},
                   f"prefill_ffn_{index}_rank_{rank}" if phase == "decode" else None)
export("output", "output", Output(model), {"hidden": decode_samples["output", 0]})
(directory / "model.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
header = [
    f"constexpr size_t hiddenSize = {config.hidden_size};",
    f"constexpr size_t layers = {config.num_hidden_layers};",
    f"constexpr size_t kvHeads = {config.num_key_value_heads // args.parts};",
    f"constexpr size_t parts = {args.parts};",
    f"constexpr size_t headSize = {head_dim};",
    f"constexpr size_t vocabulary = {config.vocab_size};",
    f"constexpr size_t prefillLength = {args.prefill_len};",
    f"constexpr size_t cacheLength = {args.max_cache_len};",
    f"constexpr size_t workspaceBytes = {metadata['workspace_bytes']};",
]
declarations = {
    "embedding": "Hidden *, Floats *, Tokens *",
    "attention": "AttentionResult *, Floats *, Bytes *, Hidden *, Cache *, Cache *, Positions *",
    "ffn": "Hidden *, Floats *, Bytes *, Hidden *",
    "output": "Hidden *, Floats *, Bytes *, Hidden *",
}
for entry in metadata["stages"]:
    name, kind = entry["name"], entry["kind"]
    header.append(f'extern "C" void _mlir_ciface_forward_{name}({declarations[kind]});')
    parameter = '{"%s", %d, %d}' % (entry["parameters"], entry["f32_elements"], entry["weight_bytes"])
    header.append(f'const {kind.capitalize()}Entry {name} = {{{parameter}, _mlir_ciface_forward_{name}}};')
for phase in ("prefill", "decode"):
    for kind in ("attention", "ffn"):
        rows = ["{" + ", ".join(f"{phase}_{kind}_{index}_rank_{rank}" for index in range(config.num_hidden_layers)) + "}"
                for rank in range(args.parts)]
        header.append(f"const {kind.capitalize()}Entry {phase}_{kind}[parts][layers] = {{" + ", ".join(rows) + "};")
(directory / "qwen-parameters.h").write_text("\n".join(header) + "\n")
