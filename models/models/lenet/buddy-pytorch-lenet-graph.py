from pathlib import Path

import torch
from buddy.compiler.frontend import DynamoCompiler
from stack.compiler.frontend.ops import tosa

from model import LeNet

model = LeNet().eval()
model.load_state_dict(
    torch.load(
        Path(__file__).resolve().parent / "output/lenet/model.pth", weights_only=True
    )
)

compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    verbose=True,
    verbose_path=Path(__file__).resolve().parent / "output/buddy-graph.txt",
    trace=None,
)
with torch.no_grad():
    graphs = compiler.importer(model, torch.zeros((1, 1, 28, 28)))

# Include Buddy node names and frontend IR in the graph inspection output.
for graph in graphs:
    graph.lower_to_top_level_ir()
