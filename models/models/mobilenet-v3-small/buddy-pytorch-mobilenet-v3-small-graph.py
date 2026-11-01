from importlib import import_module
from pathlib import Path
import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from torch._decomp import get_decompositions

output = Path(__file__).resolve().parent / "output"
output.mkdir(exist_ok=True)
mobilenet = import_module("models.models.mobilenet-v3-small.model")
model = mobilenet.load()

compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=get_decompositions(
        [torch.ops.aten.hardsigmoid.default, torch.ops.aten.hardswish.default]
    ),
    verbose=True,
    verbose_path=output / "buddy-graph.txt",
    trace=None,
)
with torch.no_grad():
    graphs = compiler.importer(model, *mobilenet.inputs())
for graph in graphs:
    graph.lower_to_top_level_ir()
