from importlib import import_module
from pathlib import Path

import numpy as np
import torch

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)

lenet = import_module("models.models.lenet.model")
model = lenet.load()
compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    trace=None,
)

with torch.no_grad():
    graphs = compiler.importer(model, *lenet.inputs())
assert len(graphs) == 1
graph = graphs[0]

params = compiler.imported_params[graph]
graph.dump(output / "buddy-graph.txt")

# Group the graph for code generation without inserting trace operations.
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
