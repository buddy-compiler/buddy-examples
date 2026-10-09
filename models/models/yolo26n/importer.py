from importlib import import_module


#!/usr/bin/env python3
# ===- import-yolo26n.py ------------------------------------------------------
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
import sys

import torch
import torch._inductor.lowering
from torch._inductor.decomposition import decompositions as inductor_decomp
from ultralytics import YOLO





import argparse

parser = argparse.ArgumentParser(description="yolo26n model AOT importer")
parser.add_argument(
    "--calibration-images", type=Path, nargs="+", required=True,
    help="Ordered calibration images, separate from inference test images.",
)
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
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--compiler-build", type=Path, required=True)
parser.add_argument("--design", required=True)
import_module(".configs.importer-param", __package__).add_arguments(parser)
args = parser.parse_args()
args.compiler_build = args.compiler_build.resolve()
sys.path.insert(0, str(args.compiler_build.resolve() / "python_packages"))
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from buddy.compiler.trace import TraceConfig, load_trace_config
from buddy.compiler.graph.transform.quantization.normalize import fold_batch_norms

design = import_module(args.design)
quant = import_module(design.__name__ + ".quant.quantize")
design_dir = Path(design.__file__).parent
torch.set_num_threads(4)
calibration_images = [path.resolve(strict=True) for path in args.calibration_images]

output_dir = Path(args.output_dir).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
os.chdir(output_dir)
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

model_path = args.checkpoint.resolve(strict=True)
model = YOLO(str(model_path)).model.eval()
fold_batch_norms(model)
detect_head = model.model[-1]
detect_head.end2end = True
detect_head.export = True
detect_head.xyxy = True

torch.set_num_threads(4)
input_tensor, calibration = quant.prepare(model, calibration_images, args.img_size, model_path, output_dir)

dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward",
    verbose=verbose,
    verbose_path=verbose_path,
    trace=trace,
)

with torch.no_grad():
    # Warm up once to initialize Detect anchors/strides for the fixed input shape.
    model(input_tensor)
    graphs = dynamo_compiler.importer(model, input_tensor)

assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]

graph.fuse_ops([simply_fuse])


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


quant.quantize(
    graph,
    params,
    _param_names(model, params),
    output_dir,
    "yolo26",
    calibration,
    rax_pack=args.compiler_build / "bin/rax-pack",
)
driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()
with open(output_dir / "subgraph0.mlir", "w") as module_file:
    print(driver.subgraphs[0]._imported_module, file=module_file)

with open(output_dir / "forward.mlir", "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)

import json
metadata = {"version": 1, "chip": design.chip, "model": "yolo26"}
(output_dir / "model.json").write_text(json.dumps(metadata))
