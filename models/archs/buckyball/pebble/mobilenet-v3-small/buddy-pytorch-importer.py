from importlib import import_module
from pathlib import Path
import json

import numpy as np
from PIL import Image

import torch
from torch._decomp import get_decompositions
from torch.nn.utils.fusion import fuse_conv_bn_eval

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

# Use the same single image and FP32 NHWC normalization as the Linux runner.
pixels = np.asarray(
    Image.open(
        model_dir.parents[3]
        / "models"
        / "mobilenet-v3-small/images/dog-32bit_224x224.bmp"
    ).convert("RGB"),
    dtype=np.float32,
)
assert pixels.shape == (224, 224, 3)
pixels /= np.float32(255.0)
pixels -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
pixels /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
pixels.tofile(output / "inputs.data")

# ===----------------------------------------------------------------------===
# Step 1: Import the graph of the checkpoint, every batch normalization
# folded into the convolution before it.
# ===----------------------------------------------------------------------===
mobilenet = import_module("models.models.mobilenet-v3-small.model")
model = mobilenet.load()
for block in model.modules():
    if (
        isinstance(block, torch.nn.Sequential)
        and isinstance(block[0], torch.nn.Conv2d)
        and isinstance(block[1], torch.nn.BatchNorm2d)
    ):
        block[0] = fuse_conv_bn_eval(block[0], block[1])
        block[1] = torch.nn.Identity()
# The traced nodes are named as in the imported graph.
compiler = DynamoCompiler(
    primary_registry={**tosa.ops_registry, **quant.ops_registry},
    aot_autograd_decomposition=get_decompositions(
        [torch.ops.aten.hardsigmoid.default, torch.ops.aten.hardswish.default]
    ),
    trace=TraceConfig(load_trace_config(model_dir / "trace/trace.toml")),
)
with torch.no_grad():
    (graph,) = compiler.importer(model, *mobilenet.inputs())
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
calibration = json.loads((output / "calibration.json").read_text())
params = quantize(graph, params, parameter_names, calibration)
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
