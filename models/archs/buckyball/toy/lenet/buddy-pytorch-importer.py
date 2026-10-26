from pathlib import Path

import numpy as np

import torch
from models.models.lenet.model import LeNet

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from buddy.compiler.trace import TraceConfig, load_trace_config

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
    primary_registry=tosa.ops_registry,
    trace=TraceConfig(load_trace_config(model_dir / "trace/trace.toml")),
)
with torch.no_grad():
    (graph,) = compiler.importer(model, torch.zeros(1, 1, 28, 28))
params = compiler.imported_params[graph]

# ===----------------------------------------------------------------------===
# Step 2: Export the original graph and FP32 parameters.
# ===----------------------------------------------------------------------===
# The toy core does not recognize any operation yet: the FP32 graph runs in
# software in its checkpoint layout, so there is nothing to permute or to
# quantize.
graph.dump(output / "01-import.txt")
# graph.dump_parameters(output / "01-import-parameters")

# ===----------------------------------------------------------------------===
# Step 3: Partition the graph: one toy core runs all of it.
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
