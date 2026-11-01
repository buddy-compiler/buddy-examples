from importlib import import_module
from pathlib import Path

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions

laya = import_module("models.models.laya.model")

compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=decompositions,
    verbose=True,
    verbose_path=Path(__file__).resolve().parent / "output/buddy-graph.txt",
    trace=None,
)
with torch.no_grad():
    graphs = compiler.importer(laya.load(), *laya.inputs())
for graph in graphs:
    graph.lower_to_top_level_ir()
