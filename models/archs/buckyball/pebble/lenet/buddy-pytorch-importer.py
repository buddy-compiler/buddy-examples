from pathlib import Path
import json

import numpy as np

import torch
from models.models.lenet.model import LeNet

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from buddy.compiler.trace import TraceConfig, load_trace_config
from stack.compiler.frontend import quant
from frontend.permute.graph import permute
from frontend.quant.graph import quantize

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
model_source = model_dir.parents[3] / "models/lenet"

# ===----------------------------------------------------------------------===
# Step 1: Import the graph of the checkpoint.
# ===----------------------------------------------------------------------===
model = LeNet().eval()
model.load_state_dict(
    torch.load(model_source / "output/lenet/model.pth", weights_only=True)
)
# The traced nodes are named as in the imported graph.
compiler = DynamoCompiler(
    primary_registry={**tosa.ops_registry, **quant.ops_registry},
    trace=TraceConfig(load_trace_config(model_dir / "trace/trace.toml")),
)
with torch.no_grad():
    (graph,) = compiler.importer(model, torch.zeros(1, 1, 28, 28))
params = compiler.imported_params[graph]
names = {parameter.data_ptr(): name for name, parameter in model.named_parameters()}
parameter_names = {
    node.name: names[value.data_ptr()]
    for node, value in zip(graph.params, params, strict=True)
}

# ===----------------------------------------------------------------------===
# Step 2.1: Export the original graph and FP32 parameters.
# ===----------------------------------------------------------------------===
graph.dump(output / "01-import.txt")
# graph.dump_parameters(output / "01-import-parameters")

# ===----------------------------------------------------------------------===
# Step 2.2: Align the layouts with the pebble Balls.
# ===----------------------------------------------------------------------===
params = permute(graph, params)
graph.dump(output / "02-permuted.txt")
# graph.dump_parameters(output / "02-permuted-parameters")

# ===----------------------------------------------------------------------===
# Step 2.3: Quantize the convolutions and Linears to INT8.
# ===----------------------------------------------------------------------===
# The activation scales come from buddy-pytorch-calibrate.py.
maxima = json.loads((output / "calibration.json").read_text())
params = quantize(graph, params, parameter_names, maxima)
graph.dump(output / "03-quanted.txt")
# graph.dump_parameters(output / "03-quanted-parameters")

# ===----------------------------------------------------------------------===
# Step 3: Partition the graph: one pebble core runs all of it.
# ===----------------------------------------------------------------------===
graph.fuse_ops([simply_fuse])
graph.dump(output / "04-partitioned.txt")
driver = GraphDriver(graph)
for index, subgraph in enumerate(driver.subgraphs):
    subgraph.lower_to_top_level_ir(buffer_layout="contiguous", writable=False)
    (output / f"subgraph{index}.mlir").write_text(str(subgraph._imported_module))
(output / "forward.mlir").write_text(str(driver.construct_main_graph(True)))
for index, dtype in enumerate((torch.float32, torch.int32, torch.int8)):
    np.concatenate(
        [p.detach().numpy().reshape(-1) for p in params if p.dtype == dtype]
    ).tofile(output / f"arg{index}.data")
