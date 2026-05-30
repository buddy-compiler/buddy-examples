# ===- buddy-lenet-import.py ---------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# This is the LeNet model AOT importer.
#
# ===---------------------------------------------------------------------------

import os
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import torch._inductor.lowering

import numpy as np
import torch
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from model import LeNet

# Retrieve the LeNet model path from environment variables.
model_path = os.environ.get("LENET_MODEL_PATH")
if model_path is None:
    raise EnvironmentError(
        "The environment variable 'LENET_MODEL_PATH' is not set or is invalid."
    )
model_dir = Path(model_path)


def load_trace_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"trace config not found: {path}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    items = data.get("trace")
    if not isinstance(items, list):
        raise ValueError("trace config must contain [[trace]] entries")

    result = {}
    ids = set()
    for item in items:
        node = item.get("node")
        trace_id = item.get("id")
        tag = item.get("tag")
        layout = item.get("layout")
        if not isinstance(node, str) or not node:
            raise ValueError("trace.node must be a non-empty string")
        if not isinstance(trace_id, int):
            raise ValueError(f"trace id for {node} must be an integer")
        if not isinstance(tag, str) or not tag:
            raise ValueError(f"trace tag for {node} must be a non-empty string")
        if not isinstance(layout, str) or not layout:
            raise ValueError(f"trace layout for {node} must be a non-empty string")
        if node in result:
            raise ValueError(f"duplicate trace node: {node}")
        if trace_id in ids:
            raise ValueError(f"duplicate trace id: {trace_id}")
        ids.add(trace_id)
        result[node] = {"id": trace_id, "tag": tag, "layout": layout}
    return result

model = LeNet()
trace_dir = model_dir / "trace"
output_dir = model_dir / "output"

model = torch.load(model_dir / "lenet-model.pth", weights_only=False)
model = model.eval()
trace_config = load_trace_config(trace_dir / "trace.toml")

# Initialize Dynamo Compiler with specific configurations as an importer.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    trace_config=trace_config,
    trace_dump_dir=output_dir,
)

data = torch.randn([1, 1, 28, 28])
# Import the model into MLIR module and parameters.
with torch.no_grad():
    graphs = dynamo_compiler.importer(model, data)


assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
pattern_list = [simply_fuse]
graphs[0].fuse_ops(pattern_list)
driver = GraphDriver(graphs[0])
driver.subgraphs[0].lower_to_top_level_ir()
path_prefix = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(path_prefix, "subgraph0.mlir"), "w") as module_file:
    print(driver.subgraphs[0]._imported_module, file=module_file)
with open(os.path.join(path_prefix, "forward.mlir"), "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)

params = dynamo_compiler.imported_params[graph]
current_path = os.path.dirname(os.path.abspath(__file__))

float32_param = np.concatenate(
    [param.detach().numpy().reshape([-1]) for param in params]
)

float32_param.tofile(Path(current_path) / "arg0.data")
