# ===- import-bert.py ----------------------------------------------------------
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
# This is the test of BERT model.
#
# ===---------------------------------------------------------------------------

import os
import argparse
from pathlib import Path
import numpy as np
import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from buddy.compiler.trace import TraceConfig, load_trace_config
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import BertForSequenceClassification, BertTokenizer

# Parse command-line arguments
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
parser.add_argument(
    "--core-count",
    type=int,
    default=1,
    help="Split the fused BERT graph into this many homogeneous Core subgraphs.",
)
args = parser.parse_args()

if args.core_count < 1:
    parser.error("--core-count must be positive")

# Ensure output directory exists
output_dir = Path(args.output_dir).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
model_dir = Path(__file__).resolve().parent

if args.trace:
    trace = TraceConfig(load_trace_config(model_dir / "trace" / "trace.toml"))
    verbose = False
    verbose_path = None
else:
    trace = None
    verbose = True
    verbose_path = os.path.join(output_dir, "output", "buddy-graph.txt")
    if os.path.exists(verbose_path):
        os.remove(verbose_path)

model = BertForSequenceClassification.from_pretrained(
    "bhadresh-savani/bert-base-uncased-emotion"
)
model.eval()
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    verbose=verbose,
    verbose_path=verbose_path,
    trace=trace,
)

tokenizer = BertTokenizer.from_pretrained("bhadresh-savani/bert-base-uncased-emotion")
inputs = {
    "input_ids": torch.tensor([[1 for _ in range(5)]], dtype=torch.int64),
    "token_type_ids": torch.tensor([[0 for _ in range(5)]], dtype=torch.int64),
    "attention_mask": torch.tensor([[1 for _ in range(5)]], dtype=torch.int64),
}
with torch.no_grad():
    graphs = dynamo_compiler.importer(model, **inputs)

assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
pattern_list = [simply_fuse]
graphs[0].fuse_ops(pattern_list)
graph = graphs[0]
fused_ops = graph.op_groups.pop("subgraph0")
if args.core_count > len(fused_ops):
    raise ValueError(
        f"cannot split {len(fused_ops)} fused BERT operations across "
        f"{args.core_count} Cores"
    )

# A homogeneous tile uses one Core compiler target. Partition its one fused
# graph into ordered chunks so each Core receives a distinct part of the model.
base, remainder = divmod(len(fused_ops), args.core_count)
offset = 0
for core_id in range(args.core_count):
    chunk_size = base + (1 if core_id < remainder else 0)
    name = f"subgraph{core_id}"
    graph.op_groups[name] = fused_ops[offset : offset + chunk_size]
    graph.group_map_device[name] = graph.group_map_device.get(
        "subgraph0", graph.device
    )
    offset += chunk_size

driver = GraphDriver(graph)
for subgraph in driver.subgraphs:
    subgraph.lower_to_top_level_ir()

# Write the MLIR module and forward graph to the specified output directory
for core_id, subgraph in enumerate(driver.subgraphs):
    with open(os.path.join(output_dir, f"subgraph{core_id}.mlir"), "w") as module_file:
        print(subgraph._imported_module, file=module_file)
with open(os.path.join(output_dir, "forward.mlir"), "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)

params = dynamo_compiler.imported_params[graph]

float32_param = np.concatenate(
    [param.detach().numpy().reshape([-1]) for param in params[:-1]]
)
float32_param.tofile(Path(output_dir) / "arg0.data")

int64_param = params[-1].detach().numpy().reshape([-1])
int64_param.tofile(Path(output_dir) / "arg1.data")
