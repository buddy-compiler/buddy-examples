from pathlib import Path

import numpy as np

import torch
from torch._decomp import get_decompositions
from torch._inductor.decomposition import decompositions as inductor_decomp
from ultralytics import YOLO

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from buddy.compiler.trace import TraceConfig, load_trace_config

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
model_source = model_dir.parents[3] / "models/yolo26n"

# ===----------------------------------------------------------------------===
# Step 1: Import the graph of the checkpoint, every batch normalization
# folded into the convolution before it and the Detect head returning its
# 300 best boxes as [x1, y1, x2, y2, score, class].
# ===----------------------------------------------------------------------===
model = YOLO(model_source / "output/yolo26n/yolo26n.pt").model.eval()
model.fuse()
head = model.model[-1]
head.end2end = True
head.export = True
head.xyxy = True
# The traced nodes are named as in the imported graph.
compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    # Splits become slices: Buddy has no lowering of split_with_sizes.
    aot_autograd_decomposition={
        **inductor_decomp,
        **get_decompositions([torch.ops.aten.split_with_sizes.default]),
    },
    trace=TraceConfig(load_trace_config(model_dir / "trace/trace.toml")),
)
image = torch.zeros(1, 3, 640, 640)
with torch.no_grad():
    # One run fixes the anchors and strides of the Detect head for 640x640.
    model(image)
    (graph,) = compiler.importer(model, image)
params = compiler.imported_params[graph]
# The anchors and strides enter the graph as the inputs after the image; as
# constants of the 640x640 input they become parameters.
for node, value in zip(graph.inputs[1:], (head.anchors, head.strides), strict=True):
    index = graph.body.index(node)
    graph._inputs.remove(index)
    graph._fake_params.append(index)
    params.append(value)

# ===----------------------------------------------------------------------===
# Step 2: Export the original graph and FP32 parameters.
#
# The model stays in FP32 and runs in software: the pebble core lowers
# every recognized convolution into fully unrolled Ball instructions, and the
# 2.7 GMAC of YOLO26n at 640x640 are far beyond what the compiler handles
# (MobileNetV3-Small, 56 MMAC, already takes 2.5 GB of MLIR).
# ===----------------------------------------------------------------------===
graph.dump(output / "01-import.txt")
# graph.dump_parameters(output / "01-import-parameters")

# ===----------------------------------------------------------------------===
# Step 3: Partition the graph: one pebble core runs all of it.
# ===----------------------------------------------------------------------===
graph.fuse_ops([simply_fuse])
graph.dump(output / "02-partitioned.txt")
driver = GraphDriver(graph)
for index, subgraph in enumerate(driver.subgraphs):
    subgraph.lower_to_top_level_ir(buffer_layout="contiguous", writable=False)
    (output / f"subgraph{index}.mlir").write_text(str(subgraph._imported_module))
(output / "forward.mlir").write_text(str(driver.construct_main_graph(True)))
np.concatenate([p.detach().numpy().reshape(-1) for p in params]).tofile(
    output / "arg0.data"
)
