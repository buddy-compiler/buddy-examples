from importlib import import_module

# ===- importer.py ----------------------------------------------------------
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
# BERT model AOT importer.
#
# ===---------------------------------------------------------------------------

import os
import sys
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import BertForSequenceClassification, BertTokenizer

# Parse command-line arguments
import argparse
add_arguments = import_module(".configs.importer-param", __package__).add_arguments

parser = argparse.ArgumentParser(description="BERT model AOT importer")
parser.add_argument(
    "--output-dir", type=str, default="./", help="Directory to save output files"
)
parser.add_argument(
    "--trace",
    action="store_true",
    default=False,
    help="Import with trace/trace.toml.",
)
parser.add_argument("--compiler-build", type=Path, required=True)
parser.add_argument("--design", required=True)
add_arguments(parser)
args = parser.parse_args()
args.compiler_build = args.compiler_build.resolve()
sys.path.insert(0, str(args.compiler_build.resolve() / "python_packages"))
design = import_module(args.design)
quant = import_module(design.__name__ + ".quant.quantize")
design_dir = Path(design.__file__).parent

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from buddy.compiler.trace import TraceConfig, load_trace_config



if args.partition_count < 1:
    parser.error("--partition-count must be positive")

# Ensure output directory exists
output_dir = Path(args.output_dir).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
model_dir = Path(__file__).resolve().parent

if args.trace:
    trace = TraceConfig(load_trace_config(design_dir / "trace" / "trace.toml"))
    verbose = False
    verbose_path = None
else:
    trace = None
    verbose = True
    verbose_path = os.path.join(output_dir, "output", "buddy-graph.txt")
    if os.path.exists(verbose_path):
        os.remove(verbose_path)

model = BertForSequenceClassification.from_pretrained(
    "bhadresh-savani/bert-base-uncased-emotion", attn_implementation="eager"
)
model.eval()
model_config = model.config

class BertInference(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self.bert = source.bert
        self.classifier = source.classifier

    def forward(self, input_ids, token_type_ids, attention_mask, position_ids):
        hidden = self.bert.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        mask = (1.0 - attention_mask[:, None, None, :].to(hidden.dtype)) * torch.finfo(hidden.dtype).min
        hidden = self.bert.encoder(hidden, attention_mask=mask)[0]
        return self.classifier(self.bert.pooler(hidden))

model = BertInference(model).eval()
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    verbose=verbose,
    verbose_path=verbose_path,
    trace=trace,
)

tokenizer = BertTokenizer.from_pretrained("bhadresh-savani/bert-base-uncased-emotion")
inputs, calibration = quant.prepare(model, tokenizer, args.sequence_length)
with torch.no_grad():
    graphs = dynamo_compiler.importer(model, **inputs)

assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
parameter_names = {parameter.data_ptr(): name for name, parameter in model.named_parameters()}
input_nodes = {
    value.data_ptr(): index
    for value, index in zip(graph._runtime_inputs_ref, graph._inputs)
}
graph._inputs = [input_nodes[value.data_ptr()] for value in inputs.values()]
graph._runtime_inputs_ref = list(inputs.values())
pattern_list = [simply_fuse]
graphs[0].fuse_ops(pattern_list)
graph = graphs[0]
metadata = {
    "chip": design.chip,
    "version": 1,
    "model": "bhadresh-savani/bert-base-uncased-emotion",
    "sequence_length": args.sequence_length,
    "num_labels": model_config.num_labels,
    "labels": [model_config.id2label[index] for index in range(model_config.num_labels)],
    "vocab_size": model_config.vocab_size,
    "parameter_sha256": {
        name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, value in model.named_parameters()
    },
}
quant.quantize(
    graph, params, [parameter_names[parameter.data_ptr()] for parameter in params],
    output_dir, "bert", calibration, args.compiler_build / "bin/rax-pack",
    assets={"model.json": json.dumps(metadata, sort_keys=True).encode()},
)
fused_ops = graph.op_groups.pop("subgraph0")
body_order = {node.name: index for index, node in enumerate(graph.body)}
fused_ops.sort(key=lambda node: body_order[node.name])
op_order = {node.name: index for index, node in enumerate(fused_ops)}
cut_points = set(range(1, len(fused_ops) + 1))
for index, node in enumerate(fused_ops):
    if isinstance(node.tensor_meta.get("dtype"), (tuple, list)):
        end = max([index] + [op_order[name] for name in node._children if name in op_order])
        cut_points.difference_update(range(index + 1, end + 1))
if args.partition_count > len(fused_ops):
    raise ValueError(
        f"cannot split {len(fused_ops)} fused BERT operations across "
        f"{args.partition_count} partitions"
    )

# Compile ordered subgraphs for one core type; the runtime selects each core.
base, remainder = divmod(len(fused_ops), args.partition_count)
offset = 0
for partition_id in range(args.partition_count):
    chunk_size = base + (1 if partition_id < remainder else 0)
    name = f"subgraph{partition_id}"
    end = len(fused_ops) if partition_id + 1 == args.partition_count else offset + chunk_size
    while end < len(fused_ops) and end not in cut_points:
        end += 1
    if end > len(fused_ops) - (args.partition_count - partition_id - 1):
        raise ValueError("cannot partition BERT without splitting a tuple result")
    graph.op_groups[name] = fused_ops[offset:end]
    graph.group_map_device[name] = graph.group_map_device.get(
        "subgraph0", graph.device
    )
    offset = end

driver = GraphDriver(graph)
for subgraph in driver.subgraphs:
    subgraph.lower_to_top_level_ir()

# Write the MLIR module and forward graph to the specified output directory
for partition_id, subgraph in enumerate(driver.subgraphs):
    with open(os.path.join(output_dir, f"subgraph{partition_id}.mlir"), "w") as module_file:
        print(subgraph._imported_module, file=module_file)
with open(os.path.join(output_dir, "forward.mlir"), "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)

params = dynamo_compiler.imported_params[graph]

float_count = sum(param.numel() for param in params if param.dtype == torch.float32)
int8_count = sum(param.numel() for param in params if param.dtype == torch.int8)
if any(param.dtype not in (torch.float32, torch.int8) for param in params):
    raise ValueError("unexpected parameter dtype in quantized BERT")
(output_dir / "bert-parameters.h").write_text(
    f"#define BERT_F32_ELEMENTS {float_count}\n"
    f"#define BERT_I8_ELEMENTS {int8_count}\n"
    f"#define BERT_SEQUENCE_LENGTH {args.sequence_length}\n"
    f"#define BERT_NUM_LABELS {model_config.num_labels}\n"
)
