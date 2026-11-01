from importlib import import_module
from pathlib import Path

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions

pi05 = import_module("models.models.pi05-base.model")

compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=decompositions,
    verbose=True,
    verbose_path=Path(__file__).resolve().parent / "output/buddy-graph.txt",
    trace=None,
)
with torch.no_grad():
    graphs = compiler.importer(pi05.load(), *pi05.inputs())

# Include Buddy node names and frontend IR in the graph inspection output.
for graph in graphs:
    graph.lower_to_top_level_ir()
