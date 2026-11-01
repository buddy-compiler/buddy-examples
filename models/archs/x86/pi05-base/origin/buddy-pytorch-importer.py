# ===- buddy-pytorch-importer.py ------------------------------------------
#
# pi0.5 AOT importer: the whole policy as one graph, plus a test case of the
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

pi05 = import_module("models.models.pi05-base.model")
model = pi05.load()

# The test case of the runner: fixed random inputs and the actions PyTorch
# samples from them.
example = pi05.inputs()
torch.manual_seed(0)
images = torch.rand(example[0].shape) * 2 - 1
tokens = torch.randint(0, 1000, example[1].shape)
masks = example[2]
noise = torch.randn(example[3].shape)
with torch.no_grad():
    actions = model(images, tokens, masks, noise)
testcase = output / "testcase"
testcase.mkdir(exist_ok=True)
images.numpy().tofile(testcase / "images.data")
tokens.numpy().tofile(testcase / "tokens.data")
masks.numpy().tofile(testcase / "masks.data")
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
# FP32 weights, then the INT64 patch positions of SigLIP.
np.concatenate(
    [p.detach().cpu().numpy().reshape(-1) for p in params if p.dtype == torch.float32]
).tofile(output / "arg0.data")
np.concatenate(
    [p.detach().cpu().numpy().reshape(-1) for p in params if p.dtype == torch.int64]
).tofile(output / "arg1.data")
