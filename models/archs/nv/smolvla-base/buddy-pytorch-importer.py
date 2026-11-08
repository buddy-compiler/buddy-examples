# ===- buddy-pytorch-importer.py ------------------------------------------
#
# SmolVLA AOT importer: the whole policy as one graph, plus a test case of the
# runner sampled by PyTorch.
#
# ===---------------------------------------------------------------------------

from importlib import import_module
from pathlib import Path
import json

import numpy as np
import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from compiler.frontend import quant
from torch._inductor.decomposition import decompositions
from frontend.permute.graph import permute
from frontend.quant.graph import quantize

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)

smolvla = import_module("models.models.smolvla-base.model")
model = smolvla.load()
# The vision encoder attends through explicit products the quantization can
# read as INT8, not through one fused attention.
model.flow.vlm_with_expert.get_vlm_model().vision_model.config._attn_implementation = (
    "eager"
)

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
    primary_registry={**tosa.ops_registry, **quant.ops_registry},
    aot_autograd_decomposition=decompositions,
    trace=None,
)
with torch.no_grad():
    (graph,) = compiler.importer(model, *example)
params = compiler.imported_params[graph]
graph.dump(output / "01-import.txt")
# Absorb the transposes PyTorch puts on the Linear weights.
permute(graph)
graph.dump(output / "02-permuted.txt")

# ===----------------------------------------------------------------------===
# Step 2.3: Quantize every matrix product: the Linears and the attention
# products to MXINT8, the patch convolution to INT8.
# ===----------------------------------------------------------------------===
# The activation scales are computed online; buddy-pytorch-calibrate.py
# accepts the quantized policy first.
calibration = json.loads((output / "calibration.json").read_text())
print(f"Quantized actions accepted at relative L2 error {calibration['error']:.4f}")
params = quantize(graph, params)
graph.dump(output / "03-quanted.txt")
graph.fuse_ops([simply_fuse])
driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()
(output / "subgraph0.mlir").write_text(str(driver.subgraphs[0]._imported_module))
(output / "forward.mlir").write_text(str(driver.construct_main_graph(True)))
# forward takes the parameters grouped by type: FP32 (the biases among
# them), INT8 weights (MXINT8 codes and exponents for the Linears).
for index, dtype in enumerate((torch.float32, torch.int8)):
    np.concatenate(
        [p.detach().cpu().numpy().reshape(-1) for p in params if p.dtype == dtype]
    ).tofile(output / f"arg{index}.data")
