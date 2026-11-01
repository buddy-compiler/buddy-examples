from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from torch._inductor.decomposition import decompositions

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from buddy.compiler.trace import TraceConfig, load_trace_config

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(exist_ok=True)
gemma4 = import_module("models.models.gemma4-e2b-it.model")

compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=decompositions,
    trace=TraceConfig(load_trace_config(model_dir / "trace/trace.toml")),
)
# One token per step: the origin lowering takes static shapes only.
with torch.no_grad():
    (graph,) = compiler.importer(gemma4.load(), *gemma4.inputs(1))
params = compiler.imported_params[graph]
graph.fuse_ops([simply_fuse])
driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()
(output / "subgraph0.mlir").write_text(str(driver.subgraphs[0]._imported_module))
(output / "forward.mlir").write_text(str(driver.construct_main_graph(True)))
np.concatenate([p.detach().cpu().float().numpy().reshape(-1) for p in params]).tofile(
    output / "arg0.data"
)
tokenizer = AutoTokenizer.from_pretrained(gemma4.checkpoint)
with (output / "vocab.txt").open("w") as stream:
    for token, index in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]):
        stream.write(token.replace("\n", "\\n") + "\n")
