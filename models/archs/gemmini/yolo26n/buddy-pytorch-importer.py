from pathlib import Path

import numpy as np
import torch
from torch._decomp import get_decompositions

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from stack.compiler.frontend.ops import tosa

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
from ultralytics import YOLO

model = YOLO(str(model_dir / "../../../models/yolo26n/output/yolo26n/yolo26n.pt")).model
head = model.model[-1]
head.end2end = True
head.export = True
head.xyxy = True
data = torch.zeros((1, 3, 640, 640))
model.eval()
compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=get_decompositions(
        [torch.ops.aten.detach.default, torch.ops.aten.fill.Tensor]
    ),
    trace=None,
)
with torch.no_grad():
    model(data)
    graphs = compiler.importer(model, data)
assert len(graphs) == 1
graph = graphs[0]
params = compiler.imported_params[graph]
graph.dump(output / "01-import.txt")
# graph.dump_parameters(output / "01-import-parameters")
graph.fuse_ops([simply_fuse])
driver = GraphDriver(graph)
for subgraph in driver.subgraphs:
    subgraph.trace(model_dir / "trace/trace.toml")
graph.dump(output / "04-traced.txt")
for index, subgraph in enumerate(driver.subgraphs):
    subgraph.lower_to_top_level_ir()
    (output / f"subgraph{index}.mlir").write_text(str(subgraph._imported_module))
(output / "forward.mlir").write_text(str(driver.construct_main_graph(True)))
parameters = np.concatenate(
    [parameter.detach().cpu().numpy().reshape(-1) for parameter in params]
)
parameters.astype(np.float32).tofile(output / "arg0.data")
(output / "labels.txt").write_text(
    "\n".join(model.names[index] for index in range(len(model.names))) + "\n"
)
