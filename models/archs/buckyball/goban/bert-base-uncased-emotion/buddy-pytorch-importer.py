from stack.compiler.frontend.trace import trace
from frontend.partition import partition
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
import json
import tomllib
from pathlib import Path
import torch
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import BertForSequenceClassification, BertTokenizer

# Parse command-line arguments
import argparse

parser = argparse.ArgumentParser(description="BERT model AOT importer")
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--compiler-build", type=Path, required=True)
parser.add_argument("--design-pb", type=Path, required=True)
parser.add_argument("--partition-count", type=int, required=True)
parser.add_argument("--sequence-length", type=int, required=True)
parser.add_argument("--repo-root", type=Path, required=True)
args = parser.parse_args()

args.compiler_build = args.compiler_build.resolve()
import __init__ as recipe
from stack.compiler.frontend.quant.importer import quantize_model_graph
from stack.compiler.frontend.quant.offline.activations import calibrate_layer_ranges
from quant.calibration import EmotionCalibration

recipe_dir = Path(__file__).parent

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from stack.compiler.frontend.ops import tosa

if args.partition_count < 1:
    parser.error("--partition-count must be positive")

# Ensure output directory exists
output_dir = Path(args.output_dir).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
model_dir = Path(__file__).resolve().parent

verbose = False
verbose_path = None

model = BertForSequenceClassification.from_pretrained(
    model_dir
    / "../../../../models/bert-base-uncased-emotion/output/bert-base-uncased-emotion",
    attn_implementation="eager",
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
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        mask = (1.0 - attention_mask[:, None, None, :].to(hidden.dtype)) * torch.finfo(
            hidden.dtype
        ).min
        hidden = self.bert.encoder(hidden, attention_mask=mask)[0]
        return self.classifier(self.bert.pooler(hidden))


model = BertInference(model).eval()
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    verbose=verbose,
    verbose_path=verbose_path,
    trace=None,
)

dataset = EmotionCalibration(args.sequence_length)
tokenizer = dataset.tokenizer
config = tomllib.loads((model_dir / "quant/quant.toml").read_text())
calibration = calibrate_layer_ranges(model, dataset, **config["calibration"])
inputs = {name: value[None] for name, value in dataset[0][0].items()}
with torch.no_grad():
    graphs = dynamo_compiler.importer(model, **inputs)

assert len(graphs) == 1
graph = graphs[0]
graph.order_inputs(
    inputs[name]
    for name in ("input_ids", "token_type_ids", "attention_mask", "position_ids")
)
params = dynamo_compiler.imported_params[graph]
parameter_names = {
    parameter.data_ptr(): name for name, parameter in model.named_parameters()
}
graph.order_inputs(inputs.values())
graph._runtime_inputs_ref = list(inputs.values())
graph = graphs[0]
quantize_model_graph(
    graph,
    params,
    [parameter_names[parameter.data_ptr()] for parameter in params],
    output_dir,
    "bert",
    calibration,
)
graph.dump(output_dir / "03-quanted.txt")
graph.fuse_ops([simply_fuse])
partition(graph, args.partition_count)
graph.dump(output_dir / "04-fusion.txt")
trace(graph, Path(__file__).parent / "trace/trace.toml")
graph.dump(output_dir / "05-traced.txt")

driver = GraphDriver(graph)
for subgraph in driver.subgraphs:
    subgraph.lower_to_top_level_ir()

# Write the MLIR module and forward graph to the specified output directory
for partition_id, subgraph in enumerate(driver.subgraphs):
    with open(
        os.path.join(output_dir, f"subgraph{partition_id}.mlir"), "w"
    ) as module_file:
        print(subgraph._imported_module, file=module_file)
forward = str(driver.construct_main_graph(True))
(Path(output_dir) / "forward.mlir").write_text(forward)


(output_dir / "params.f32").replace(output_dir / "arg0.data")
(output_dir / "weights.bin").replace(output_dir / "arg1.data")
(output_dir / "vocab.txt").write_text(
    "\n".join(
        token
        for token, _ in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])
    )
    + "\n"
)
