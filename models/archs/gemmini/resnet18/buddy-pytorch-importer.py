from pathlib import Path

import numpy as np
import torch

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from stack.compiler.frontend.ops import tosa

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=None)
model.load_state_dict(
    torch.load(
        model_dir / "../../../models/resnet18/output/resnet18/model.pth",
        weights_only=True,
    )
)
for layer in model.modules():
    if isinstance(layer, torch.nn.ReLU):
        layer.inplace = False
data = torch.zeros((1, 3, 224, 224))
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
(output / "Labels.txt").write_text(
    "\n".join(ResNet18_Weights.DEFAULT.meta["categories"]) + "\n"
)
