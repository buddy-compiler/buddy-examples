# ===- buddy-pytorch-importer.py -------------------------------------------
#
# MobileNetV3-Small on toy: the toy hart lowers supported INT8 contractions to VecBall. The
# graph folds batch normalization into convolutions before quantization. In <output>/:
#   subgraph0.mlir, forward.mlir  the graph and its entry
#   arg0.data                     the FP32 parameters (arg1/arg2 hold INT32 biases and INT8 weights)
#   inputs.data                   the baremetal image, normalized, NCHW
#   reference.data                the logits of the quantized PyTorch model
#
#   buddy-pytorch-importer.py <output>
#
# ===---------------------------------------------------------------------===

from importlib import import_module
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from torch._decomp import get_decompositions
import json
from quantized_model import folded, reference
from frontend.quant.graph import quantize
from frontend.permute.graph import permute
from compiler.frontend import quant

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa

output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=True)
mobilenet = import_module("models.models.mobilenet-v3-small.model")
model = folded()
maxima = json.loads((output / "calibration.json").read_text())

# The baremetal image, normalized as dataset.cpp normalizes its pixels.
pixels = np.asarray(
    Image.open(
        Path(mobilenet.__file__).parent / "images/dog-32bit_224x224.bmp"
    ).convert("RGB"),
    dtype=np.uint8,
)
values = pixels.astype(np.float32) / np.float32(255)
values = (values - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
    [0.229, 0.224, 0.225], np.float32
)
image = np.ascontiguousarray(values.transpose(2, 0, 1))
image.tofile(output / "inputs.data")
with torch.no_grad():
    reference(maxima)(torch.from_numpy(image)[None]).numpy().astype(np.float32).tofile(
        output / "reference.data"
    )

compiler = DynamoCompiler(
    primary_registry={**tosa.ops_registry, **quant.ops_registry},
    aot_autograd_decomposition=get_decompositions(
        [torch.ops.aten.hardsigmoid.default, torch.ops.aten.hardswish.default]
    ),
)
with torch.no_grad():
    (graph,) = compiler.importer(model, *mobilenet.inputs())
params = compiler.imported_params[graph]
graph.dump(output / "01-import.txt")
names = {value.data_ptr(): name for name, value in model.named_parameters()}
parameter_names = {
    node.name: names[value.data_ptr()]
    for node, value in zip(graph.params, params, strict=True)
    if value.data_ptr() in names
}
permute(graph)
graph.dump(output / "02-permuted.txt")
params = quantize(graph, params, parameter_names, maxima)
graph.dump(output / "03-quanted.txt")
graph.fuse_ops([simply_fuse])
driver = GraphDriver(graph)
(subgraph,) = driver.subgraphs
subgraph.lower_to_top_level_ir(buffer_layout="contiguous", writable=False)
(output / "subgraph0.mlir").write_text(str(subgraph._imported_module))
(output / "forward.mlir").write_text(str(driver.construct_main_graph(True)))
for index, dtype in enumerate((torch.float32, torch.int32, torch.int8)):
    np.concatenate(
        [p.detach().numpy().reshape(-1) for p in params if p.dtype == dtype]
    ).tofile(output / f"arg{index}.data")
