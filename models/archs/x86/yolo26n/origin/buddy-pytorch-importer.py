from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from buddy.compiler.trace import TraceConfig, load_trace_config
from torch._decomp import get_decompositions

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)

yolo = import_module("models.models.yolo26n.model")
model = yolo.load()
# Export boxes and class scores; the runner selects and filters detections.
model.model[-1].postprocess = lambda predictions: predictions
compiler = DynamoCompiler(
    aot_autograd_decomposition=get_decompositions(
        [
            torch.ops.aten.silu.default,
            torch.ops.aten.detach.default,
            torch.ops.aten.fill.Tensor,
        ]
    ),
    primary_registry=tosa.ops_registry,
    trace=TraceConfig(load_trace_config(model_dir / "trace/trace.toml")),
)
with torch.no_grad():
    graphs = compiler.importer(model, *yolo.inputs())
assert len(graphs) == 1
graph = graphs[0]
params = compiler.imported_params[graph]
# The anchors and strides enter the graph as the inputs after the image; as
# constants of the 640x640 input they become parameters.
head = model.model[-1]
for node, value in zip(graph.inputs[1:], (head.anchors, head.strides), strict=True):
    index = graph.body.index(node)
    graph._inputs.remove(index)
    graph._fake_params.append(index)
    params.append(value)
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
