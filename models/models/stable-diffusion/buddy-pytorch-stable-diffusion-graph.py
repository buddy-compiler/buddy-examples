from importlib import import_module
from pathlib import Path

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions

output = Path(__file__).resolve().parent / "output"

for name in ("text_encoder", "unet", "vae"):
    module = import_module(f"models.models.stable-diffusion.model-{name}")
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=decompositions,
        verbose=True,
        verbose_path=output / f"buddy-graph-{name}.txt",
        trace=None,
    )
    with torch.no_grad():
        graphs = compiler.importer(module.load(), *module.inputs())
    for graph in graphs:
        graph.lower_to_top_level_ir()
(output / "buddy-graph.txt").write_text(
    "\n".join(
        (output / f"buddy-graph-{name}.txt").read_text()
        for name in ("text_encoder", "unet", "vae")
    )
)
