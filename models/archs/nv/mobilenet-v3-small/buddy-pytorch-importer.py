from importlib import import_module
from pathlib import Path
import json

import numpy as np
import torch
from torch._decomp import get_decompositions

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from buddy.compiler.trace import TraceConfig, load_trace_config
from compiler.frontend import quant
from frontend.permute.graph import permute
from frontend.quant.graph import quantize

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)

mobilenet = import_module("models.models.mobilenet-v3-small.model")
model = mobilenet.load()
compiler = DynamoCompiler(
    aot_autograd_decomposition=get_decompositions(
        [torch.ops.aten.hardsigmoid.default, torch.ops.aten.hardswish.default]
    ),
    primary_registry={**tosa.ops_registry, **quant.ops_registry},
    trace=TraceConfig(load_trace_config(model_dir / "trace/trace.toml")),
)
with torch.no_grad():
    graphs = compiler.importer(model, *mobilenet.inputs())
assert len(graphs) == 1
graph = graphs[0]
params = compiler.imported_params[graph]
names = {
    value.data_ptr(): name
    for name, value in [*model.named_parameters(), *model.named_buffers()]
}
parameter_names = {
    node.name: names[value.data_ptr()]
    for node, value in zip(graph.params, params, strict=True)
}
graph.dump(output / "01-import.txt")
# The convolutions run in NHWC; the transposes around the Linears are absorbed.
permute(graph)
graph.dump(output / "02-permuted.txt")

# ===----------------------------------------------------------------------===
# Step 2.3: Quantize every matrix product to INT8 (W8A8): the weights with one
# scale per output channel, the activations with one calibrated scale.
# ===----------------------------------------------------------------------===
# The activation scales come from buddy-pytorch-calibrate.py.
calibration = json.loads((output / "calibration.json").read_text())
params = quantize(graph, params, parameter_names, calibration)
graph.dump(output / "03-quanted.txt")

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
