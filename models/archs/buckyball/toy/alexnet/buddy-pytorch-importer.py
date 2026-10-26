from stack.compiler.frontend.transforms import quantize
from pathlib import Path
import tomllib

import numpy as np
import torch
from torchvision.models import alexnet

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from stack.compiler.frontend.ops import tosa

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
model_source = model_dir.parents[3] / "models/alexnet"
model = alexnet(weights=None).eval()
model.load_state_dict(
    torch.load(model_source / "output/alexnet/model.pth", weights_only=True)
)
for layer in model.modules():
    if isinstance(layer, torch.nn.ReLU):
        layer.inplace = False
compiler = DynamoCompiler(primary_registry=tosa.ops_registry, trace=None)
with torch.no_grad():
    (graph,) = compiler.importer(model, torch.zeros(1, 3, 224, 224))
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
# Step 2.2: Normalize the logical graph before target parameter packing.
# ===----------------------------------------------------------------------===
graph.dump(output / "02-permuted.txt")
# graph.dump_parameters(output / "02-permuted-parameters")

# ===----------------------------------------------------------------------===
# Step 2.3: Quantize weights offline and activations online on the CPU.
# ===----------------------------------------------------------------------===
config = tomllib.loads((model_dir / "quant/quant.toml").read_text())
params = quantize(graph, config, parameter_names=parameter_names)
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
# Step 3: Lower the physical graph and export the model entry point.
# ===----------------------------------------------------------------------===
for index, subgraph in enumerate(driver.subgraphs):
    subgraph.lower_to_top_level_ir()
    (output / f"subgraph{index}.mlir").write_text(str(subgraph._imported_module))
(output / "forward.mlir").write_text(str(driver.construct_main_graph(True)))
for index, dtype in enumerate((torch.float32, torch.int8)):
    np.concatenate(
        [p.detach().cpu().numpy().reshape(-1) for p in params if p.dtype == dtype]
    ).tofile(output / f"arg{index}.data")
