#!/usr/bin/env python3
# ===- import-qwen3.py ---------------------------------------------------
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
# 
#
# ===---------------------------------------------------------------------------

import os
import argparse
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    StaticCache,
)
from torch._inductor.decomposition import decompositions as inductor_decomp
import numpy

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import (
    simply_fuse,
)
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.graph.type import TensorDType
from buddy.compiler.trace import TraceConfig, load_trace_config

# Add argument parser to allow custom output directory.
parser = argparse.ArgumentParser(description="Qwen3-0.6B Model AOT Importer")
parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Directory to save output files. Defaults to the model directory.",
)
parser.add_argument(
    "--precision",
    type=str,
    default="f32",
    choices=["f32"],
    help="Precision mode for generated MLIR and input data. Choose from 'f32'.",
)
parser.add_argument(
    "--trace",
    action="store_true",
    default=False,
    help="Import with trace/prefill.toml and trace/decode.toml.",
)
parser.add_argument(
    "--max-cache-len",
    type=int,
    default=1024,
    help="Static cache length for generated prefill/decode MLIR.",
)
parser.add_argument(
    "--prefill-len",
    type=int,
    default=1024,
    help="Prefill input token length for generated MLIR.",
)
args = parser.parse_args()
if args.prefill_len < 1:
    raise ValueError("--prefill-len must be positive")
if args.max_cache_len < args.prefill_len:
    raise ValueError("--max-cache-len must be greater than or equal to --prefill-len")

model_dir = Path(__file__).resolve().parent
output_dir = Path(args.output_dir).resolve() if args.output_dir else model_dir
output_dir.mkdir(parents=True, exist_ok=True)
if args.trace:
    trace_prefill = TraceConfig(
        load_trace_config(model_dir / "trace" / "prefill.toml")
    )
    trace_decode = TraceConfig(
        load_trace_config(model_dir / "trace" / "decode.toml")
    )
    verbose = False
    verbose_path_prefill = None
    verbose_path_decode = None
else:
    trace_prefill = None
    trace_decode = None
    verbose = True
    verbose_path_prefill = os.path.join(
        output_dir, "output", "buddy-graph-prefill.txt"
    )
    verbose_path_decode = os.path.join(
        output_dir, "output", "buddy-graph-decode.txt"
    )
    for path in [verbose_path_prefill, verbose_path_decode]:
        if os.path.exists(path):
            os.remove(path)

# Retrieve the Qwen3-0.6B model path from environment variables.
model_path = os.environ.get("QWEN3_0_6B_MODEL_PATH")
if model_path is None:
    model_path = "Qwen/Qwen3-0.6B"

# Initialize the model from the specified model path.
model = AutoModelForCausalLM.from_pretrained(model_path).eval()
model.config.use_cache = False

# Initialize Dynamo Compiler with specific configurations as an importer.

dynamo_compiler_prefill = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward_prefill",
    verbose=verbose,
    verbose_path=verbose_path_prefill,
    trace=trace_prefill,
)

dynamo_compiler_decode = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward_decode",
    verbose=verbose,
    verbose_path=verbose_path_decode,
    trace=trace_decode,
)

# Import the model into MLIR module and parameters.
with torch.no_grad():
    past_key_values_prefill = StaticCache(
        config=model.config, max_cache_len=args.max_cache_len
    )
    past_key_values_decode = StaticCache(
        config=model.config, max_cache_len=args.max_cache_len
    )

    data_prefill = {
        "input_ids": torch.zeros((1, args.prefill_len), dtype=torch.int64),
    }
    data_decode = {
        "input_ids": torch.zeros((1, 1), dtype=torch.int64),
    }

    cache_position = torch.tensor([200], dtype=torch.int64)

    graphs_prefill = dynamo_compiler_prefill.importer(
        model,
        input_ids=data_prefill["input_ids"],
        past_key_values=past_key_values_prefill,
        use_cache=True,
    )
    # Initialize past_key_values once during the first forward call
    model(
        input_ids=data_decode["input_ids"],
        past_key_values=past_key_values_decode,
        use_cache=True,
    )

    graphs_decode = dynamo_compiler_decode.importer(
        model,
        input_ids=data_decode["input_ids"],
        use_cache=True,
        cache_position=cache_position,
        past_key_values=past_key_values_decode,
    )

assert len(graphs_prefill) == 1
assert len(graphs_decode) == 1
graph_prefill = graphs_prefill[0]
graph_decode = graphs_decode[0]

params = dynamo_compiler_prefill.imported_params[graph_prefill]
param_meta = graph_prefill.params_shapes
if len(params) != len(param_meta):
    raise ValueError(
        f"Qwen3 parameter count mismatch: {len(params)} params vs "
        f"{len(param_meta)} metadata entries"
    )
pattern_list_prefill = [
    simply_fuse,
]
pattern_list_decode = [
    simply_fuse,
]

graphs_prefill[0].fuse_ops(pattern_list_prefill)
graphs_decode[0].fuse_ops(pattern_list_decode)

graph_prefill.op_groups["subgraph0_prefill"] = graph_prefill.op_groups.pop(
    "subgraph0"
)
graph_prefill.group_map_device["subgraph0_prefill"] = DeviceType.CPU

graph_decode.op_groups["subgraph0_decode"] = graph_decode.op_groups.pop(
    "subgraph0"
)
graph_decode.group_map_device["subgraph0_decode"] = DeviceType.CPU

driver_prefill = GraphDriver(graphs_prefill[0])
driver_prefill.subgraphs[0].lower_to_top_level_ir()

driver_decode = GraphDriver(graphs_decode[0])
driver_decode.subgraphs[0].lower_to_top_level_ir()

def tensor_to_numpy(param):
    param = param.detach().cpu().contiguous()
    if param.dtype == torch.bfloat16:
        return param.view(torch.uint16).numpy().reshape([-1])
    if param.dtype == torch.float32:
        return param.numpy().reshape([-1])
    raise TypeError(f"Unsupported Qwen3 parameter dtype: {param.dtype}")


def export_param_pack(dtype, filename):
    packed = [
        tensor_to_numpy(param)
        for param, meta in zip(params, param_meta)
        if meta.dtype == dtype
    ]
    if not packed:
        raise ValueError(f"No Qwen3 parameters found for dtype {dtype}")
    numpy.concatenate(packed).tofile(os.path.join(output_dir, filename))


def export_subgraph(module, filename, func_name):
    text = str(module)
    private_needle = f"func.func private @{func_name}"
    public_needle = f"func.func @{func_name}"
    if private_needle in text:
        text = text.replace(private_needle, public_needle, 1)
    elif public_needle not in text:
        raise ValueError(
            f"Qwen3 subgraph module missing expected symbol: {func_name}"
        )
    with open(os.path.join(output_dir, filename), "w") as module_file:
        print(text, file=module_file)


for dtype in sorted({meta.dtype for meta in param_meta}, key=str):
    if dtype == TensorDType.BFloat16:
        export_param_pack(dtype, "arg0_0_6b.data")
    elif dtype == TensorDType.Float32:
        export_param_pack(dtype, "arg1_0_6b.data")
    else:
        raise TypeError(f"Unsupported Qwen3 parameter pack dtype: {dtype}")

export_subgraph(
    driver_prefill.subgraphs[0]._imported_module,
    "subgraph0_prefill_0_6b.mlir",
    "subgraph0_prefill",
)

with open(
    os.path.join(output_dir, "forward_prefill_0_6b.mlir"), "w"
) as module_file:
    print(driver_prefill.construct_main_graph(True), file=module_file)

export_subgraph(
    driver_decode.subgraphs[0]._imported_module,
    "subgraph0_decode_0_6b.mlir",
    "subgraph0_decode",
)
with open(
    os.path.join(output_dir, "forward_decode_0_6b.mlir"), "w"
) as module_file:
    print(driver_decode.construct_main_graph(True), file=module_file)
