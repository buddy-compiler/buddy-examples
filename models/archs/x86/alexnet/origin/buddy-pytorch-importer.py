from importlib import import_module
from pathlib import Path
import numpy as np
import torch
from torchvision.models import AlexNet_Weights
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
alexnet = import_module("models.models.alexnet.model")
model = alexnet.load()
compiler = DynamoCompiler(primary_registry=tosa.ops_registry, trace=None)
with torch.no_grad():
    (graph,) = compiler.importer(model, *alexnet.inputs())
params = compiler.imported_params[graph]
graph.dump(output / "01-import.txt")
graph.fuse_ops([simply_fuse])
graph.dump(output / "05-fusion.txt")
driver = GraphDriver(graph)
for index, subgraph in enumerate(driver.subgraphs):
    subgraph.lower_to_top_level_ir()
    (output / f"subgraph{index}.mlir").write_text(str(subgraph._imported_module))
(output / "forward.mlir").write_text(str(driver.construct_main_graph(True)))
(output / "labels.txt").write_text(
    "\n".join(AlexNet_Weights.IMAGENET1K_V1.meta["categories"]) + "\n"
)
np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in params]).astype(
    np.float32
).tofile(output / "arg0.data")
