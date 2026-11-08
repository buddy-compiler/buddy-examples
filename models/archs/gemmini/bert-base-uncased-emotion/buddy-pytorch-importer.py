from importlib import import_module
from pathlib import Path

import numpy as np
import json

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from compiler.frontend import quant
from frontend.permute.graph import permute
from frontend.quant.graph import quantize
from torch._inductor.decomposition import decompositions

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)

# ===----------------------------------------------------------------------===
# Step 1: Import the classifier as one graph.
# ===----------------------------------------------------------------------===
bert = import_module("models.models.bert-base-uncased-emotion.model")
model = bert.load()
compiler = DynamoCompiler(
    # Gemmini has no conversion hardware: the CPU converts between FP32 and
    # the INT8 codes.
    primary_registry={**tosa.ops_registry, **quant.ref_registry},
    aot_autograd_decomposition=decompositions,
    trace=None,
)
with torch.no_grad():
    (graph,) = compiler.importer(model, *bert.inputs())
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

# ===----------------------------------------------------------------------===
# Step 2.2: Absorb the transposes PyTorch puts on the Linear weights.
# ===----------------------------------------------------------------------===
permute(graph)
graph.dump(output / "02-permuted.txt")

# ===----------------------------------------------------------------------===
# Step 3: Quantize the Linears and the attention products to INT8: Gemmini
# computes in INT8 only.
# ===----------------------------------------------------------------------===
# The activation scales come from buddy-pytorch-calibrate.py.
calibration = json.loads((output / "calibration.json").read_text())
params = quantize(graph, params, parameter_names, calibration)
graph.dump(output / "03-quanted.txt")

# ===----------------------------------------------------------------------===
# Step 4: Lower the graph for gemmini.
# ===----------------------------------------------------------------------===
graph.fuse_ops([simply_fuse])
driver = GraphDriver(graph)
for index, subgraph in enumerate(driver.subgraphs):
    subgraph.lower_to_top_level_ir()
    (output / f"subgraph{index}.mlir").write_text(str(subgraph._imported_module))
(output / "forward.mlir").write_text(str(driver.construct_main_graph(True)))
# forward takes the parameters grouped by type: FP32, INT32 biases, INT8
# weights.
for index, dtype in enumerate((torch.float32, torch.int32, torch.int8)):
    np.concatenate(
        [p.detach().numpy().reshape(-1) for p in params if p.dtype == dtype]
    ).tofile(output / f"arg{index}.data")
