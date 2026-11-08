from importlib import import_module
from pathlib import Path
import json

import numpy as np

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch._inductor.decomposition import decompositions

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from compiler.frontend import quant
from frontend.permute.graph import permute
from frontend.quant.graph import quantize, smooth_inputs

torch.set_num_threads(4)

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)

# ===----------------------------------------------------------------------===
# Step 1: Import the complete checkpoint model.
# ===----------------------------------------------------------------------===
laya = import_module("models.models.laya.model")
decisions = import_module("models.models.laya.data")
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
inputs = laya.inputs()
with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
    (graph,) = compiler.importer(model, *inputs)
graph.order_inputs(inputs)
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
graph.fuse_ops([simply_fuse])
driver = GraphDriver(graph)
for index, subgraph in enumerate(driver.subgraphs):
    subgraph.lower_to_top_level_ir(buffer_layout="contiguous", writable=False)
    (output / f"subgraph{index}.mlir").write_text(str(subgraph._imported_module))
(output / "forward.mlir").write_text(str(driver.construct_main_graph(True)))
# The parameters grouped by type: FP32, INT32 biases, INT8 weights.
for index, dtype in enumerate((torch.float32, torch.int32, torch.int8)):
    np.concatenate(
        [p.detach().numpy().reshape(-1) for p in params if p.dtype == dtype]
    ).tofile(output / f"arg{index}.data")

# One held-out typed decision, shared by baremetal and Linux.
sample, labels = decisions.load(64, 1)
sample = tuple(value[:1] for value in sample)
(output / "input.data").write_bytes(
    b"".join(value.numpy().tobytes() for value in sample)
)
from frontend.reference import evaluate

with sdpa_kernel(SDPBackend.MATH):
    expected = evaluate(model, calibration, sample)
np.concatenate([value.numpy().reshape(-1) for value in expected]).tofile(
    output / "expected.data"
)
(output / "reference.json").write_text(
    json.dumps(
        {
            "choice": expected[0].argmax(-1).item(),
            "action": expected[1].argmax(-1).item(),
            "label": labels[0].item(),
        }
    )
    + "\n"
)
