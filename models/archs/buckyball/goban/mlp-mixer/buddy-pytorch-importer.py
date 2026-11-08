from importlib import import_module
from pathlib import Path
import json

import numpy as np
from PIL import Image

import torch
from torch._inductor.decomposition import decompositions

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from buddy.compiler.trace import TraceConfig, load_trace_config
from compiler.frontend import quant
from frontend.permute.graph import permute
from frontend.quant.graph import quantize, smooth_norms

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
model_source = model_dir.parents[3] / "models"

# ===----------------------------------------------------------------------===
# Step 1: Import the graph of the checkpoint.
# ===----------------------------------------------------------------------===
mixer = import_module("models.models.mlp-mixer.model")
model = mixer.load()
# The activation scales come from buddy-pytorch-calibrate.py; the smoothing
# of the norms changes the weights before the import.
calibration = json.loads((output / "calibration.json").read_text())
smooth_norms(model, calibration["smooth"])
# The traced nodes are named as in the imported graph.
compiler = DynamoCompiler(
    primary_registry={**tosa.ops_registry, **quant.ops_registry},
    aot_autograd_decomposition=decompositions,
    trace=TraceConfig(load_trace_config(model_dir / "trace/trace.toml")),
)
with torch.no_grad():
    (graph,) = compiler.importer(model, *mixer.inputs())
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
# Step 2.2: Absorb the transposes around the Linears and run the convolutions
# in NHWC.
# ===----------------------------------------------------------------------===
permute(graph)
graph.dump(output / "02-permuted.txt")

# ===----------------------------------------------------------------------===
# Step 2.3: Quantize every Linear to INT8: the goban matmul thread computes in
# INT8 only.
# ===----------------------------------------------------------------------===
params = quantize(graph, params, parameter_names, calibration)
graph.dump(output / "03-quanted.txt")

# ===----------------------------------------------------------------------===
# Step 3: Partition the graph: one goban core runs all of it.
# ===----------------------------------------------------------------------===
graph.fuse_ops([simply_fuse])
graph.dump(output / "04-partitioned.txt")
driver = GraphDriver(graph)
for index, subgraph in enumerate(driver.subgraphs):
    subgraph.lower_to_top_level_ir(buffer_layout="contiguous", writable=False)
    (output / f"subgraph{index}.mlir").write_text(str(subgraph._imported_module))
(output / "forward.mlir").write_text(str(driver.construct_main_graph(True)))
# forward takes the parameters grouped by type: FP32, INT32 biases, INT8
# weights.
for index, dtype in enumerate((torch.float32, torch.int32, torch.int8)):
    np.concatenate(
        [p.detach().numpy().reshape(-1) for p in params if p.dtype == dtype]
    ).tofile(output / f"arg{index}.data")

# ===----------------------------------------------------------------------===
# Step 4: The sample input: dog.bmp resized to 256, center-cropped to 224,
# scaled to [-1, 1] and cut into 196 patches of [C, 16, 16].
# ===----------------------------------------------------------------------===
image = Image.open(model_source / "alexnet/images/dog.bmp").convert("RGB")
width, height = image.size
size = (256, 256 * height // width) if width <= height else (256 * width // height, 256)
image = image.resize(size, Image.Resampling.BICUBIC)
left, top = round((image.width - 224) / 2), round((image.height - 224) / 2)
image = image.crop((left, top, left + 224, top + 224))
pixels = np.asarray(image, dtype=np.float32) / np.float32(127.5) - np.float32(1.0)
patches = (
    pixels.transpose(2, 0, 1)
    .reshape(3, 14, 16, 14, 16)
    .transpose(1, 3, 0, 2, 4)
    .reshape(196, 768)
)
np.ascontiguousarray(patches).tofile(output / "input.f32")
