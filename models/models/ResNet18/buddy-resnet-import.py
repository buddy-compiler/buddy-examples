# ===- buddy-resnet-import.py --------------------------------------------------
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
# This is the ResNet18 model AOT importer.
#
# ===---------------------------------------------------------------------------

import os
import argparse
from pathlib import Path
import sys
import torch
import torchvision.models as models
import torch._inductor.lowering
from torch._inductor.decomposition import decompositions as inductor_decomp
from torch._decomp import remove_decompositions

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from buddy.compiler.trace import TraceConfig, load_trace_config

from framework.quant.core.importer import quantize_model_graph

# Parse command-line arguments
parser = argparse.ArgumentParser(description="ResNet18 model AOT importer")
parser.add_argument(
    "--output-dir", type=str, default="./", help="Directory to save output files."
)
parser.add_argument(
    "--trace",
    action="store_true",
    default=False,
    help="Import with trace/trace.toml.",
)
parser.add_argument(
    "--trace-config",
    type=Path,
    default=Path("trace/trace.toml"),
    help="Trace config relative to the model directory.",
)
args = parser.parse_args()

# Ensure output directory exists
output_dir = Path(args.output_dir).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
model_dir = Path(__file__).resolve().parent
if args.trace:
    trace = TraceConfig(load_trace_config(model_dir / args.trace_config))
    verbose = False
    verbose_path = None
else:
    trace = None
    verbose = True
    verbose_path = os.path.join(output_dir, "output", "buddy-graph.txt")
    if os.path.exists(verbose_path):
        os.remove(verbose_path)

# Retrieve the ResNet18 model path.
model_path = str(model_dir)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = model.eval()

# Remove the num_batches_tracked attribute.
for layer in model.modules():
    if isinstance(layer, torch.nn.BatchNorm2d):
        if hasattr(layer, "num_batches_tracked"):
            del layer.num_batches_tracked

DEFAULT_DECOMPOSITIONS = [
    torch.ops.aten.max_pool2d_with_indices.default,
]

remove_decompositions(inductor_decomp, DEFAULT_DECOMPOSITIONS)

# Initialize Dynamo Compiler with specific configurations as an importer.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    verbose=verbose,
    verbose_path=verbose_path,
    trace=trace,
)
data = torch.randn([1, 3, 224, 224])
# Import the model into MLIR module and parameters.
with torch.no_grad():
    graphs = dynamo_compiler.importer(model, data)
assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
pattern_list = [simply_fuse]
graphs[0].fuse_ops(pattern_list)
quantize_model_graph(
    graph,
    params,
    [name for name, _ in model.named_parameters()]
    + [name for name, _ in model.named_buffers()],
    output_dir,
    "resnet18",
    False,
)
driver = GraphDriver(graphs[0])
driver.subgraphs[0].lower_to_top_level_ir()

# Write the MLIR module and forward graph to the specified output directory
with open(output_dir / "subgraph0.mlir", "w") as module_file:
    print(driver.subgraphs[0]._imported_module, file=module_file)
with open(output_dir / "forward.mlir", "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)
