from stack.compiler.frontend.trace import trace
from importlib import import_module

#!/usr/bin/env python3
# ===- buddy-pytorch-importer.py ------------------------------------------------------
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
# This is the yolo26n model AOT importer.
#
# ===---------------------------------------------------------------------------

import json
import os
from pathlib import Path

import torch
import torch._inductor.lowering
from torch._inductor.decomposition import decompositions as inductor_decomp
from ultralytics import YOLO


import argparse

parser = argparse.ArgumentParser(description="yolo26n model AOT importer")
parser.add_argument(
    "--output-dir",
    type=str,
    default="./",
    help="Directory to save output files.",
)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--compiler-build", type=Path, required=True)
parser.add_argument("--img-size", type=int, required=True)
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
import tomllib

recipe_dir = Path(__file__).parent

output_dir = Path(args.output_dir).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
os.chdir(output_dir)
model_dir = Path(__file__).resolve().parent

verbose = False
verbose_path = None

model_path = args.checkpoint.resolve(strict=True)
model = YOLO(str(model_path)).model.eval()
fold_batch_norms(model)
detect_head = model.model[-1]
detect_head.end2end = True
detect_head.export = True
detect_head.xyxy = True

from quant.calibration import CocoCalibration

config = tomllib.loads((model_dir / "quant/quant.toml").read_text())
calibration = calibrate_layer_ranges(
    model, CocoCalibration(args.img_size), **config["calibration"]
)
input_tensor = torch.zeros(1, 3, args.img_size, args.img_size)

dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward",
    verbose=verbose,
    verbose_path=verbose_path,
    trace=None,
)

with torch.no_grad():
    # Warm up once to initialize Detect anchors/strides for the fixed input shape.
    model(input_tensor)
    graphs = dynamo_compiler.importer(model, input_tensor)

assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]


def _param_names(mod, imported):
    state = [
        (n, t)
        for n, t in list(mod.named_parameters()) + list(mod.named_buffers())
        if not n.endswith("num_batches_tracked")
    ]
    extras = []
    dh = mod.model[-1]
    for attr in ("anchors", "strides"):
        t = getattr(dh, attr, None)
        if torch.is_tensor(t):
            extras.append((f"model.{len(mod.model)-1}.{attr}", t))
    state = state + extras
    used = set()
    names = []
    for p in imported:
        hit = None
        for n, t in state:
            if n in used:
                continue
            if t.shape != p.shape:
                continue
            if torch.equal(t.detach().cpu().float(), p.detach().cpu().float()):
                hit = n
                break
        if hit is None:
            raise ValueError(
                f"imported param shape {tuple(p.shape)} has no matching "
                "named parameter/buffer/detect anchors"
            )
        used.add(hit)
        names.append(hit)
    return names


quantize_model_graph(
    graph,
    params,
    _param_names(model, params),
    output_dir,
    "yolo26",
    calibration,
)
graph.dump(output_dir / "03-quanted.txt")
graph.fuse_ops([simply_fuse])
graph.dump(output_dir / "04-fusion.txt")
trace(graph, Path(__file__).parent / "trace/trace.toml")
graph.dump(output_dir / "05-traced.txt")
driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()
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

metadata = {"version": 1, "design": recipe.design, "model": "yolo26"}
(output_dir / "model.json").write_text(json.dumps(metadata))

(output_dir / "params.f32").replace(output_dir / "arg0.data")
(output_dir / "weights.bin").replace(output_dir / "arg1.data")
(output_dir / "labels.txt").write_text(
    "\n".join(model.names[i] for i in range(len(model.names))) + "\n"
)
