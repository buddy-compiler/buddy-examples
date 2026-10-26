from pathlib import Path

import numpy as np
import torch

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from stack.compiler.frontend.ops import tosa
from models.models.lenet.model import LeNet

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)

model = LeNet()
model.load_state_dict(
    torch.load(
        (model_dir / "../../../../models/lenet/output/lenet/model.pth").resolve(),
        weights_only=True,
    )
)
data = torch.zeros((1, 1, 28, 28))
model.eval()
compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    trace=None,
)

with torch.no_grad():
    graphs = compiler.importer(model, data)
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
