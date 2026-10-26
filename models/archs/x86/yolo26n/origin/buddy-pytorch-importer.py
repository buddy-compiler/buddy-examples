import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch._decomp import get_decompositions

parser = argparse.ArgumentParser()


parser.add_argument("--checkpoint", type=Path, required=True)
args = parser.parse_args()
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from stack.compiler.frontend.ops import tosa
from buddy.compiler.trace import TraceConfig, load_trace_config

output = Path("output").resolve()
output.mkdir(parents=True, exist_ok=True)
from ultralytics import YOLO

model = YOLO(str(args.checkpoint)).model
head = model.model[-1]
head.end2end = True
head.export = True
head.xyxy = True
# Export boxes and class scores; the runner selects and filters detections.
head.postprocess = lambda predictions: predictions
data = torch.zeros((1, 3, 640, 640))
model.eval()
compiler = DynamoCompiler(
    aot_autograd_decomposition=get_decompositions(
        [
            torch.ops.aten.silu.default,
            torch.ops.aten.detach.default,
            torch.ops.aten.fill.Tensor,
        ]
    ),
    primary_registry=tosa.ops_registry,
    trace=TraceConfig(load_trace_config(Path(__file__).parent / "trace/trace.toml")),
)
with torch.no_grad():
    # Populate detection anchors before capture so cache updates are not ABI outputs.
    model(data)
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
