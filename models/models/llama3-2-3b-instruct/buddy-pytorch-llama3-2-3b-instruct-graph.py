from importlib import import_module
from pathlib import Path

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions

llama3 = import_module("models.models.llama3-2-3b-instruct.model")

compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=decompositions,
    verbose=True,
    verbose_path=Path(__file__).resolve().parent / "output/buddy-graph.txt",
    trace=None,
)
with torch.no_grad():
    graphs = compiler.importer(llama3.load(), *llama3.inputs(1))

# Include Buddy node names and frontend IR in the graph inspection output.
for graph in graphs:
    graph.lower_to_top_level_ir()
