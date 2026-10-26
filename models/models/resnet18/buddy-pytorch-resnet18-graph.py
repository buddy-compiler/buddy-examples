from pathlib import Path
import torch
from buddy.compiler.frontend import DynamoCompiler
from stack.compiler.frontend.ops import tosa
from torch._inductor.decomposition import decompositions

output = Path(__file__).resolve().parent / "output"
output.mkdir(exist_ok=True)
from torchvision.models import resnet18

model = resnet18(weights=None).eval()
model.load_state_dict(torch.load(output / "resnet18/model.pth", weights_only=True))
for layer in model.modules():
    if hasattr(layer, "inplace"):
        layer.inplace = False
inputs = (torch.zeros((1, 3, 224, 224)),)

compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=decompositions,
    verbose=True,
    verbose_path=output / "buddy-graph.txt",
    trace=None,
)
with torch.no_grad():
    graphs = compiler.importer(model, *inputs)
for graph in graphs:
    graph.lower_to_top_level_ir()
