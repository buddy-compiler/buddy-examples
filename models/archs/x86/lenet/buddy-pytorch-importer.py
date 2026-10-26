from frontend.permute.graph import permute
from stack.compiler.frontend.transforms import quantize
from pathlib import Path
import tomllib

import numpy as np
import torch

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from stack.compiler.frontend.ops import tosa
from models.models.lenet.model import LeNet

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)


# ===----------------------------------------------------------------------===
# Step 1: Load the local checkpoint and prepare the model for graph capture.
# ===----------------------------------------------------------------------===
parameters = torch.load(
    (model_dir / "../../../models/lenet/output/lenet/model.pth").resolve(),
    weights_only=True,
)
quant_config = tomllib.loads((model_dir / "quant/quant.toml").read_text())

model = LeNet().eval()
model.load_state_dict(parameters)

data = torch.zeros((1, 1, 28, 28))


# ===----------------------------------------------------------------------===
# Step 2.1: Capture the original FP32 graph without trace.
# ===----------------------------------------------------------------------===
compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    trace=None,
)

with torch.no_grad():
    graphs = compiler.importer(model, data)
assert len(graphs) == 1
graph = graphs[0]

params = compiler.imported_params[graph]
graph.dump(output / "01-import.txt")
# graph.dump_parameters(output / "01-import-parameters")


# ===----------------------------------------------------------------------===
# Step 2.2: Normalize activation layouts while preserving checkpoint weight axes.
# ===----------------------------------------------------------------------===
names = {parameter.data_ptr(): name for name, parameter in model.named_parameters()}
parameter_names = {
    node.name: names[parameter.data_ptr()]
    for node, parameter in zip(graph.params, params, strict=True)
}
permute(graph)
graph.dump(output / "02-permuted.txt")
# graph.dump_parameters(output / "02-permuted-parameters")


# ===----------------------------------------------------------------------===
# Step 2.3: Quantize weights offline and compute activation scales online on x86.
# Absorb matrix weight transposes into the INT8 contraction.
# ===----------------------------------------------------------------------===
params = quantize(
    graph,
    quant_config,
    parameter_names=parameter_names,
)
graph.dump(output / "03-quanted.txt")
# graph.dump_parameters(output / "03-quanted-parameters")


# ===----------------------------------------------------------------------===
# Step 2.4: Keep the complete model in subgraph0 and apply its trace policy.
# ===----------------------------------------------------------------------===
graph.fuse_ops([simply_fuse])
driver = GraphDriver(graph)
for subgraph in driver.subgraphs:
    subgraph.trace(model_dir / "trace/trace.toml")
graph.dump(output / "04-traced.txt")


# ===----------------------------------------------------------------------===
# Step 3: Export subgraph0 and the entry point with packed parameter buffers.
# ===----------------------------------------------------------------------===
for index, subgraph in enumerate(driver.subgraphs):
    subgraph.lower_to_top_level_ir()
    (output / f"subgraph{index}.mlir").write_text(str(subgraph._imported_module))
(output / "forward.mlir").write_text(str(driver.construct_main_graph(True)))


# ===----------------------------------------------------------------------===
# Step 4: Export parameter buffers in ABI order.
# arg0.data contains FP32 parameters and scales; arg1.data contains INT8 weights.
# ===----------------------------------------------------------------------===
for index, dtype in enumerate((torch.float32, torch.int8)):
    parameters = np.concatenate(
        [
            parameter.detach().cpu().numpy().reshape(-1)
            for parameter in params
            if parameter.dtype == dtype
        ]
    )
    parameters.tofile(output / f"arg{index}.data")
