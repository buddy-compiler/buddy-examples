from importlib import import_module

#!/usr/bin/env python3
# ===- import-gemma4.py ---------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# This is the AOT importer for the Gemma4 model.
#
# ===---------------------------------------------------------------------------

import argparse
import sys
from pathlib import Path

import json
import tomllib
import torch
import torch._dynamo as dynamo

parser = argparse.ArgumentParser(description="Gemma4 Model AOT Importer")
parser.add_argument(
    "--output-dir",
    type=str,
    default="./",
    help="Directory to save output files.",
)
parser.add_argument(
    "--trace",
    action="store_true",
    default=False,
    help="Import with trace/trace.toml.",
)
parser.add_argument("--compiler-build", type=Path, required=True)
parser.add_argument("--trace-config", type=Path)
parser.add_argument("--design-module", required=True)
parser.add_argument("--chip", required=True)
parser.add_argument("--prefill-length", type=int, required=True)
parser.add_argument("--run-config", type=Path, required=True)
parser.add_argument("--isa-dir", type=Path, required=True)
parser.add_argument("--kernel-passes", required=True)
import_module(".configs.importer-param", __package__).add_arguments(parser)
args = parser.parse_args()
sys.path.insert(0, str(args.compiler_build.resolve() / "python_packages"))

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph.operation import *  # noqa: F403
from .abi import verify_config, verify_graph
from .native import NativeGemma
from .cache import Cache
from buddy.compiler.ops import tosa
from buddy.compiler.trace import TraceConfig, load_trace_config
from torch._inductor.decomposition import decompositions as inductor_decomp

prepare = import_module("stack.models.models.gemma4-e2b-it.prepare")
design = import_module(args.design_module)
for component in ("tasks", "entry", "linear", "partition"):
    setattr(design, component, import_module(design.__name__ + "." + component))
from .task_export import Exporter, Parameters
from .task_plan import export_phase
from .task_program import emit_program

tiles = design.partition.participants(
    tomllib.loads(args.run_config.read_text())["tile_indices"]
)
registry = dict(tosa.ops_registry)

output_dir = Path(args.output_dir).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
task_exporter = Exporter(
    args.output_dir,
    args.compiler_build,
    args.kernel_passes.split(),
    design,
    args.isa_dir,
)
model_dir = Path(__file__).resolve().parent
trace = TraceConfig(load_trace_config(args.trace_config)) if args.trace else None
verbose = False
verbose_path = None

model_path = args.checkpoint

CacheLength = 512
PrefillLength = args.prefill_length
if not 1 <= PrefillLength <= CacheLength:
    raise ValueError("Gemma prefill length must be in 1..512")

model, tc = prepare.load_model(model_path)
model.config.use_cache = False
model = NativeGemma(model)

# Initialize the quantized cache before graph capture.
print("Pre-initializing MXFP8 cache for prefill...")
quant = import_module(design.__name__ + ".quant.quantize")
cache_prefill = Cache(tc, CacheLength, quant.cache)
with torch.no_grad():
    model(
        input_ids=torch.zeros((1, 1), dtype=torch.int64),
        position_ids=torch.zeros((1, 1), dtype=torch.int64),
        past_key_values=cache_prefill,
    )
cache_prefill.reset()
verify_config(tc, cache_prefill)

dynamo.reset()

dynamo_compiler_prefill = DynamoCompiler(
    primary_registry=registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward_prefill",
    verbose=verbose,
    verbose_path=verbose_path,
    trace=trace,
)

dynamo_compiler_decode = DynamoCompiler(
    primary_registry=registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward_decode",
    verbose=verbose,
    verbose_path=verbose_path,
    trace=trace,
)

quant.cache.register(dynamo_compiler_prefill)
quant.cache.register(dynamo_compiler_decode)

with torch.no_grad():
    data_prefill = {
        "input_ids": torch.zeros((1, PrefillLength), dtype=torch.int64),
    }

    prefill_positions = torch.arange(PrefillLength, dtype=torch.int64)[None]
    graphs_prefill = dynamo_compiler_prefill.importer(
        model,
        input_ids=data_prefill["input_ids"],
        position_ids=prefill_positions,
        past_key_values=cache_prefill,
    )

    print("Pre-initializing MXFP8 cache for decode...")
    cache_decode = Cache(tc, CacheLength, quant.cache)
    model(
        input_ids=torch.zeros((1, 1), dtype=torch.int64),
        position_ids=torch.zeros((1, 1), dtype=torch.int64),
        past_key_values=cache_decode,
    )

    for layer in cache_decode.layers:
        layer.cumulative_length.fill_(200)

    dynamo.reset()

    decode_tokens = torch.zeros((1, 1), dtype=torch.int64)
    position_ids = torch.tensor([[200]], dtype=torch.int64)

    graphs_decode = dynamo_compiler_decode.importer(
        model,
        input_ids=decode_tokens,
        position_ids=position_ids,
        past_key_values=cache_decode,
    )

assert len(graphs_prefill) == 1, f"Expected 1 prefill graph, got {len(graphs_prefill)}"
assert len(graphs_decode) == 1, f"Expected 1 decode graph, got {len(graphs_decode)}"
graph_prefill = graphs_prefill[0]
graph_decode = graphs_decode[0]
for graph, tokens, cache, positions in (
    (graph_prefill, data_prefill["input_ids"], cache_prefill, [prefill_positions]),
    (graph_decode, decode_tokens, cache_decode, [position_ids]),
):
    expected = [tokens, *positions]
    for layer in cache.layers:
        expected.extend(
            (
                layer.cumulative_length,
                layer.key_codes,
                layer.key_scales,
                layer.value_codes,
                layer.value_scales,
            )
        )
    captured = {
        tensor.data_ptr(): index
        for tensor, index in zip(graph._runtime_inputs_ref, graph._inputs)
    }
    if set(captured) != {tensor.data_ptr() for tensor in expected}:
        raise ValueError(
            "Gemma captured runtime tensors do not match the native cache interface"
        )
    graph._inputs = [captured[tensor.data_ptr()] for tensor in expected]
    graph._runtime_inputs_ref = expected

contracts = {}
for phase, graph in (("prefill", graph_prefill), ("decode", graph_decode)):
    result = next(node for node in graph.body if isinstance(node, OutputOp))

    def describe(node):
        return {
            "name": node.name,
            "shape": list(node.tensor_meta["shape"]),
            "dtype": str(node.tensor_meta["dtype"]),
            "parents": list(node._parents),
            "args": [str(arg) for arg in node.args],
        }

    contracts[phase] = {
        "nodes": {
            node.name: {
                "parents": list(node._parents),
                "args": [str(arg) for arg in node.args],
            }
            for node in graph.body
        },
        "inputs": [describe(node) for node in graph.inputs],
        "outputs": [describe(graph.node_table[name]) for name in result.args],
    }
(output_dir / "capture-abi.json").write_text(json.dumps(contracts, indent=2))
verify_graph(graph_prefill, "prefill", PrefillLength)
verify_graph(graph_decode, "decode", 1)

layout = import_module(design.__name__ + ".permute.layout")
window_bytes = layout.window_bytes(args.compiler_build)
metadata = {
    "chip": args.chip,
    "model": model_path,
    "precision": "mixed-mxfp8-f32",
    "prefill_length": PrefillLength,
    "cache_length": CacheLength,
    "quantization": {
        "linear_weights": "mxfp8",
        "linear_activations": "mxfp8",
        "embedding": "mxfp8",
        "kv_cache": "mxfp8_e4m3",
        "kv_cache_scale": "e8m0",
        "kv_cache_scale_selection": "cover_maximum_f32_finite",
        "kv_cache_block_size": 32,
        "norm_and_attention": "f32",
    },
    "bank_bytes": window_bytes,
    "parameters": {},
}
constants = [
    f"inline constexpr size_t PrefillLength = {PrefillLength};",
    f"inline constexpr size_t CacheLength = {CacheLength};",
    f"inline constexpr size_t MaxVocabSize = {tc.vocab_size};",
]
for phase, graph, compiler in (
    ("prefill", graph_prefill, dynamo_compiler_prefill),
    ("decode", graph_decode, dynamo_compiler_decode),
):
    params = list(compiler.imported_params[graph])
    original_params = {node.name: value for node, value in zip(graph.params, params)}
    if any(param.dtype != torch.float32 for param in params):
        raise ValueError(f"{phase}: unsupported parameter dtype")
    quant.quantize(
        graph,
        params,
        output_dir / phase,
        phase,
        args.compiler_build / "bin/rax-pack",
        window_bytes,
    )
    parameters = Parameters(phase, graph, original_params, params, output_dir / phase)
    export_phase(
        task_exporter,
        phase,
        model.model,
        parameters,
        tiles,
        PrefillLength if phase == "prefill" else 1,
        design,
    )
    floats = sum(param.numel() for param in params if param.dtype == torch.float32)
    weights = sum(param.numel() for param in params if param.dtype == torch.int8)
    metadata["parameters"][phase] = {"f32_elements": floats, "weight_bytes": weights}
    constants.extend(
        (
            f"inline constexpr size_t {phase}Floats = {floats};",
            f"inline constexpr size_t {phase}Bytes = {weights};",
        )
    )
task_manifest = task_exporter.finish(tiles)
plan_sources, workspace_bound = emit_program(
    output_dir / "tasks", task_manifest, metadata, PrefillLength
)
with (output_dir / "tasks/CMakeLists.txt").open("a") as cmake:
    cmake.write(
        "target_sources(gemma4_kernels PRIVATE\n"
        + "".join(f"  {source}\n" for source in plan_sources)
        + ")\n"
    )
    cmake.write(
        'target_include_directories(gemma4_kernels PRIVATE "${DESIGN_DIR}" "${REPO_ROOT}/stack/runtime/include")\n'
    )
metadata["execution_tiles"] = list(tiles)
metadata["task_manifest"] = "tasks/manifest.json"
metadata["task_kernel_count"] = len(task_manifest["kernels"])
metadata["task_workspace_bound_bytes"] = workspace_bound
metadata["task_workspace_bound_kind"] = "static-liveness-first-fit"
metadata["float_parameter_bytes"] = sum(
    value["f32_elements"] * 4 for value in metadata["parameters"].values()
)
if (
    metadata["parameters"]["prefill"]["weight_bytes"]
    != metadata["parameters"]["decode"]["weight_bytes"]
):
    raise ValueError("Gemma phases must use one canonical packed weight payload")
metadata["mxfp8_parameter_bytes"] = metadata["parameters"]["prefill"]["weight_bytes"]
metadata["parameter_bytes"] = (
    metadata["float_parameter_bytes"] + metadata["mxfp8_parameter_bytes"]
)
constants.append(f"inline constexpr size_t bankBytes = {window_bytes};")
(output_dir / "model.json").write_text(json.dumps(metadata, indent=2))
(output_dir / "gemma-parameters.h").write_text(
    "#pragma once\n#include <cstddef>\n" + "\n".join(constants) + "\n"
)

prepare.export_vocabulary(model_path, output_dir / "vocab.txt")
print("All files saved to:", output_dir)
