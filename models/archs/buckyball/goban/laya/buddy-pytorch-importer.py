from importlib import import_module
from pathlib import Path
import json

import numpy as np

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch._inductor.decomposition import decompositions

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from compiler.frontend import quant
from frontend.permute.graph import permute
from frontend.quant.graph import quantize, smooth_inputs

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)

# ===----------------------------------------------------------------------===
# Step 1: Import the complete checkpoint model.
# ===----------------------------------------------------------------------===
laya = import_module("models.models.laya.model")
# The recipe runs the checkpoint fine-tuned on typed decisions.
laya.checkpoint = laya.checkpoint / "typed-decisions"
model = laya.load()
# The activation scales come from buddy-pytorch-calibrate.py; the smoothing
# changes the weights before the import.
calibration = json.loads((output / "calibration.json").read_text())
smooth_inputs(model, calibration["smooth"])
compiler = DynamoCompiler(
    primary_registry={**tosa.ops_registry, **quant.ops_registry},
    aot_autograd_decomposition=decompositions,
    trace=None,
)
# The attention of the head runs as its math: its two products are matrix
# products of the graph, like those of the encoder.
with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
    (graph,) = compiler.importer(model, *laya.inputs())
params = compiler.imported_params[graph]
names = {
    value.data_ptr(): name
    for name, value in [*model.named_parameters(), *model.named_buffers()]
}
parameter_names = {
    node.name: names[value.data_ptr()]
    for node, value in zip(graph.params, params, strict=True)
}

# ===----------------------------------------------------------------------===
# Step 2.1: Export the original graph.
# ===----------------------------------------------------------------------===
graph.dump(output / "01-import.txt")
# graph.dump_parameters(output / "01-import-parameters")

# ===----------------------------------------------------------------------===
# Step 2.2: Absorb the Linear weight transposes without changing parameters.
# ===----------------------------------------------------------------------===
permute(graph)
graph.dump(output / "02-permuted.txt")
# graph.dump_parameters(output / "02-permuted-parameters")

# ===----------------------------------------------------------------------===
# Step 2.3: Quantize every Linear and attention product to INT8: the goban
# matmul thread computes in INT8 only.
# ===----------------------------------------------------------------------===
params = quantize(graph, params, parameter_names, calibration)
graph.dump(output / "03-quanted.txt")
# The parameters grouped by type: FP32, INT32 biases, INT8 weights.
for index, dtype in enumerate((torch.float32, torch.int32, torch.int8)):
    np.concatenate(
        [p.detach().numpy().reshape(-1) for p in params if p.dtype == dtype]
    ).tofile(output / f"arg{index}.data")
