from pathlib import Path
import numpy as np
import torch
from torchvision.models import alexnet, AlexNet_Weights
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from stack.compiler.frontend.ops import tosa

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
model = alexnet(weights=None).eval()
model.load_state_dict(
    torch.load(
        model_dir / "../../../../models/alexnet/output/alexnet/model.pth",
        weights_only=True,
    )
)
for layer in model.modules():
    if isinstance(layer, torch.nn.ReLU):
        layer.inplace = False
compiler = DynamoCompiler(primary_registry=tosa.ops_registry, trace=None)
with torch.no_grad():
    (graph,) = compiler.importer(model, torch.zeros((1, 3, 224, 224)))
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
