from importlib import import_module
from pathlib import Path
import sys
import json

import numpy as np
from PIL import Image

import torch

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from buddy.compiler.trace import TraceConfig, load_trace_config
from compiler.frontend import quant
from frontend.permute.graph import permute
from frontend.quant.graph import quantize

model_dir = Path(__file__).resolve().parent
output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=True)

# One NCHW input, prepared identically to the Linux runner.
pixels = np.asarray(
    Image.open(
        Path(import_module("models.models.lenet").__file__).parent / "images/8.bmp"
    ).convert("RGB"),
    dtype=np.float32,
)
gray = (
    np.float32(0.114) * pixels[:, :, 2]
    + np.float32(0.587) * pixels[:, :, 1]
    + np.float32(0.299) * pixels[:, :, 0]
) / np.float32(255)
inputs = gray * np.float32(2) - np.float32(1)
inputs.tofile(output / "inputs.data")

# ===----------------------------------------------------------------------===
# Step 1: Import the graph of the checkpoint.
# ===----------------------------------------------------------------------===
lenet = import_module("models.models.lenet.model")
model = lenet.load()
# The traced nodes are named as in the imported graph.
compiler = DynamoCompiler(
    primary_registry={**tosa.ops_registry, **quant.ops_registry},
    trace=TraceConfig(load_trace_config(model_dir / "trace/trace.toml")),
)
with torch.no_grad():
    (graph,) = compiler.importer(model, *lenet.inputs())
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
# Step 2.2: Absorb the transposes around the Linears and run the convolutions
# in NHWC.
# ===----------------------------------------------------------------------===
permute(graph)
graph.dump(output / "02-permuted.txt")

# ===----------------------------------------------------------------------===
# Step 2.3: Quantize the convolutions and Linears to INT8.
# ===----------------------------------------------------------------------===
# The activation scales come from buddy-pytorch-calibrate.py.
maxima = json.loads((model_dir / "output/calibration.json").read_text())
params = quantize(graph, params, parameter_names, maxima)
graph.dump(output / "03-quanted.txt")

# ===----------------------------------------------------------------------===
# Step 3: Partition the graph: one toy hart runs all of it.
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

# The reference of the baremetal check: the INT8 model the calibration
# accepted (weights, biases and inputs of every convolution and Linear on
# their INT8 grids) on the same image.
reference_model = lenet.load()
for name, module in reference_model.named_modules():
    if not isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
        continue
    axes = list(range(1, module.weight.ndim))
    step = module.weight.detach().abs().amax(axes, keepdim=True) / 127
    module.weight.data = quant.quantize(module.weight.detach(), 1 / step) * step
    scale = torch.tensor(maxima[name] / 127, dtype=torch.float32)
    bias_step = step.reshape(-1) * scale
    module.bias.data = quant.quantize_bias(module.bias.detach(), bias_step) * bias_step
    module.register_forward_pre_hook(
        lambda module, inputs, scale=scale: quant.quantize(inputs[0], 1 / scale) * scale
    )
with torch.no_grad():
    reference = reference_model(torch.from_numpy(inputs).reshape(1, 1, 28, 28))
reference.numpy().astype(np.float32).tofile(output / "reference.data")

# ===----------------------------------------------------------------------===
# The dataset of the accuracy runs (dataset.cpp, from the model region): the
# MNIST test images, normalized as the image above, the logits of the same
# reference on each (the reference a sample must match) and their labels.
# ===----------------------------------------------------------------------===
from torchvision import datasets

test = datasets.MNIST(
    Path(import_module("models.models.lenet").__file__).parent / "output/datasets",
    train=False,
    download=False,
)
images = test.data.numpy().astype(np.float32) / np.float32(255) * np.float32(
    2
) - np.float32(1)
(output / "dataset").mkdir(exist_ok=True)
images.tofile(output / "dataset/inputs.data")
with torch.no_grad():
    logits = reference_model(torch.from_numpy(images).reshape(-1, 1, 28, 28))
logits.numpy().astype(np.float32).tofile(output / "dataset/reference.data")
test.targets.numpy().astype(np.int64).tofile(output / "dataset/labels.data")
