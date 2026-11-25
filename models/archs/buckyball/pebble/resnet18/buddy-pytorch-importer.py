import copy
from importlib import import_module
from pathlib import Path
import sys
import json

import numpy as np
from PIL import Image

import torch
from torch.nn.utils.fusion import fuse_conv_bn_eval

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import linalg, tosa
from buddy.compiler.trace import TraceConfig, load_trace_config
from compiler.frontend import quant
from frontend.permute.graph import permute
from frontend.quant.graph import quantize
from models.archs.buckyball.pebble.fixed import fixed
from models.archs.buckyball.pebble.fixed_reference import FixedGraph

model_dir = Path(__file__).resolve().parent
output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=True)

# Use the same single image and FP32 NHWC normalization as the Linux runner.
pixels = np.asarray(
    Image.open(
        Path(import_module("models.models.resnet18").__file__).parent
        / "images/dog-32bit_224x224.bmp"
    ).convert("RGB"),
    dtype=np.float32,
)
assert pixels.shape == (224, 224, 3)
pixels /= np.float32(255.0)
pixels -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
pixels /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
pixels.tofile(output / "inputs.data")

# ===----------------------------------------------------------------------===
# Step 1: Import the graph of the checkpoint.
# ===----------------------------------------------------------------------===
resnet18 = import_module("models.models.resnet18.model")
model = resnet18.load()
# Every BatchNorm follows a convolution in its module: fold it into the
# convolution weights and bias.
for module in model.modules():
    names = list(module._modules)
    for conv, norm in zip(names, names[1:]):
        if isinstance(module._modules[conv], torch.nn.Conv2d) and isinstance(
            module._modules[norm], torch.nn.BatchNorm2d
        ):
            module._modules[conv] = fuse_conv_bn_eval(
                module._modules[conv], module._modules[norm]
            )
            module._modules[norm] = torch.nn.Identity()
reference_model = copy.deepcopy(model)
# The traced nodes are named as in the imported graph.
compiler = DynamoCompiler(
    primary_registry={
        **tosa.ops_registry,
        **quant.ops_registry,
        "TensorConstantOp": linalg.tensor_constant_op,
    },
    trace=TraceConfig(load_trace_config(model_dir / "trace/trace.toml")),
)
with torch.no_grad():
    (graph,) = compiler.importer(model, *resnet18.inputs())
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

# ===----------------------------------------------------------------------===
# Step 2.2: Align the layouts with the pebble Balls.
# ===----------------------------------------------------------------------===
params = permute(graph, params)
graph.dump(output / "02-permuted.txt")

# ===----------------------------------------------------------------------===
# Step 2.3: Quantize the fc Linear to INT8.
# ===----------------------------------------------------------------------===
# The activation scales come from buddy-pytorch-calibrate.py.
calibration = json.loads((model_dir / "output/calibration.json").read_text())
params = quantize(graph, params, parameter_names, calibration)
graph.dump(output / "03-quanted.txt")
params = fixed(graph, params)
graph.dump(output / "04-fixed.txt")

# ===----------------------------------------------------------------------===
# Step 3: Partition the graph: one pebble hart runs all of it.
# ===----------------------------------------------------------------------===
graph.fuse_ops([simply_fuse])
graph.dump(output / "05-partitioned.txt")
driver = GraphDriver(graph)
for index, subgraph in enumerate(driver.subgraphs):
    subgraph.lower_to_top_level_ir(buffer_layout="contiguous", writable=False)
    (output / f"subgraph{index}.mlir").write_text(str(subgraph._imported_module))
(output / "forward.mlir").write_text(str(driver.construct_main_graph(True)))
for index, dtype in enumerate((torch.float32, torch.int32, torch.int8)):
    np.concatenate(
        [p.detach().numpy().reshape(-1) for p in params if p.dtype == dtype]
    ).tofile(output / f"arg{index}.data")

# The reference of the baremetal check: the logits of the fixed-point model
# the device runs, on the same image.
with torch.no_grad():
    reference = FixedGraph(reference_model, calibration).run(
        torch.from_numpy(pixels.transpose(2, 0, 1).copy())[None]
    )
reference.numpy().astype(np.float32).tofile(output / "reference.data")
