# ===- buddy-pytorch-importer.py ------------------------------------------
#
# SmolVLA AOT importer: the whole policy as one graph, plus a test case of the
# runner sampled by PyTorch.
#
# ===---------------------------------------------------------------------------

from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)

smolvla = import_module("models.models.smolvla-base.model")
model = smolvla.load()

# The test case of the runner: fixed random inputs and the actions PyTorch
# samples from them.
example = smolvla.inputs()
torch.manual_seed(0)
images = torch.rand(example[0].shape) * 2 - 1
lang_tokens = torch.randint(0, 1000, example[1].shape)
lang_masks = example[2]
state = torch.randn(example[3].shape)
noise = torch.randn(example[4].shape)
with torch.no_grad():
    actions = model(images, lang_tokens, lang_masks, state, noise)
testcase = output / "testcase"
testcase.mkdir(exist_ok=True)
images.numpy().tofile(testcase / "images.data")
lang_tokens.numpy().tofile(testcase / "lang_tokens.data")
lang_masks.numpy().tofile(testcase / "lang_masks.data")
state.numpy().tofile(testcase / "state.data")
noise.numpy().tofile(testcase / "noise.data")
actions.numpy().tofile(testcase / "actions.data")

compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=decompositions,
    trace=None,
)
with torch.no_grad():
    (graph,) = compiler.importer(model, *example)
params = compiler.imported_params[graph]
graph.fuse_ops([simply_fuse])
driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()
(output / "subgraph0.mlir").write_text(str(driver.subgraphs[0]._imported_module))
(output / "forward.mlir").write_text(str(driver.construct_main_graph(True)))
np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in params]).tofile(
    output / "arg0.data"
)
