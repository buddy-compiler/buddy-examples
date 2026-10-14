import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--repo-root", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--compiler-build", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.repo_root.resolve()))
sys.path.insert(0, str(args.compiler_build.resolve() / "python_packages"))
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from buddy.compiler.trace import TraceConfig, load_trace_config

torch.set_num_threads(4)
output = args.output_dir.resolve()
output.mkdir(parents=True, exist_ok=True)
from ultralytics import YOLO

model = YOLO(str(args.checkpoint)).model
head = model.model[-1]
head.end2end = True
head.export = True
head.xyxy = True
data = torch.zeros((1, 3, 640, 640))
model.eval()
compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    trace=TraceConfig(load_trace_config(Path(__file__).parent / "trace/trace.toml")),
)
with torch.no_grad():
    graphs = compiler.importer(model, data)
assert len(graphs) == 1
graph = graphs[0]
params = compiler.imported_params[graph]
graph.fuse_ops([simply_fuse])
driver = GraphDriver(graph)
for index, subgraph in enumerate(driver.subgraphs):
    subgraph.lower_to_top_level_ir()
    (output / f"subgraph{index}.mlir").write_text(str(subgraph._imported_module))
(output / "forward.mlir").write_text(str(driver.construct_main_graph(True)))
parameters = np.concatenate(
    [parameter.detach().cpu().numpy().reshape(-1) for parameter in params]
)
parameters.astype(np.float32).tofile(output / "arg0.data")
payload = output / "yolo26.payload"
payload.mkdir(exist_ok=True)
parameters.astype(np.float32).tofile(payload / "params.f32")
(payload / "weights.bin").write_bytes(b"")
(output / "model.json").write_text(
    json.dumps({"version": 1, "chip": "gemmini", "model": "yolo26"})
)
