from stack.compiler.frontend.trace import trace
from importlib import import_module

# ===- buddy-pytorch-importer.py --------------------------------------------------
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
from pathlib import Path
import torch
import torchvision.models as models
import torch._inductor.lowering
from torch._inductor.decomposition import decompositions as inductor_decomp
from torch._decomp import remove_decompositions

# Parse command-line arguments
import argparse

parser = argparse.ArgumentParser(description="ResNet18 model AOT importer")
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--compiler-build", type=Path, required=True)
parser.add_argument("--repo-root", type=Path, required=True)
args = parser.parse_args()

args.compiler_build = args.compiler_build.resolve()
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from stack.compiler.frontend.ops import tosa

from stack.compiler.frontend.quant.transforms.normalize import fold_batch_norms

import __init__ as recipe
from stack.compiler.frontend.quant.importer import quantize_model_graph
from stack.compiler.frontend.quant.offline.activations import calibrate_layer_ranges
from quant.calibration import ImageNetCalibration
import tomllib

recipe_dir = Path(__file__).parent

# Ensure output directory exists
output_dir = Path(args.output_dir).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
model_dir = Path(__file__).resolve().parent

verbose = False
verbose_path = None

# Retrieve the ResNet18 model path.
model_path = str(model_dir)

model = models.resnet18(weights=None)
model.load_state_dict(
    torch.load(
        model_dir / "../../../../models/resnet18/output/resnet18/model.pth",
        weights_only=True,
    )
)
model = model.eval()
for layer in model.modules():
    if isinstance(layer, torch.nn.ReLU):
        layer.inplace = False
fold_batch_norms(model)

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
    trace=None,
)
config = tomllib.loads((model_dir / "quant/quant.toml").read_text())
calibration = calibrate_layer_ranges(
    model, ImageNetCalibration(), **config["calibration"]
)
# Shape-only graph capture is independent of dataset calibration.
data = torch.zeros(1, 3, 224, 224)
# Import the model into MLIR module and parameters.
with torch.no_grad():
    graphs = dynamo_compiler.importer(model, data)
assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
quantize_model_graph(
    graph,
    params,
    [name for name, _ in model.named_parameters()]
    + [name for name, _ in model.named_buffers()],
    output_dir,
    "resnet18",
    calibration,
)
graph.dump(output_dir / "03-quanted.txt")
graph.fuse_ops([simply_fuse])
graph.dump(output_dir / "04-fusion.txt")
trace(graph, Path(__file__).parent / "trace/trace.toml")
graph.dump(output_dir / "05-traced.txt")
driver = GraphDriver(graphs[0])
driver.subgraphs[0].lower_to_top_level_ir()

# Write the MLIR module and forward graph to the specified output directory
with open(output_dir / "subgraph0.mlir", "w") as module_file:
    print(driver.subgraphs[0]._imported_module, file=module_file)
with open(output_dir / "forward.mlir", "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)

import json

(output_dir / "activation-scales.json").write_text(
    json.dumps(
        {
            name: [[value.tolist() for value in record] for record in records]
            for name, records in calibration.items()
        },
        indent=2,
    )
    + "\n"
)

metadata = {"version": 1, "design": recipe.design, "model": "resnet18"}
(output_dir / "model.json").write_text(json.dumps(metadata))

(output_dir / "params.f32").replace(output_dir / "arg0.data")
(output_dir / "weights.bin").replace(output_dir / "arg1.data")
(output_dir / "Labels.txt").write_text(
    "\n".join(models.ResNet18_Weights.DEFAULT.meta["categories"]) + "\n"
)
