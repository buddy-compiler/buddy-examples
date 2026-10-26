from frontend.permute.graph import permute
from stack.compiler.frontend.transforms import quantize
import tomllib
from quant.calibration import ImageNetCalibration
from pathlib import Path
import numpy as np
import torch
from torchvision.models import alexnet, AlexNet_Weights
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from stack.compiler.frontend.ops import tosa

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)


# ===----------------------------------------------------------------------===
# Step 1: Load the local checkpoint and prepare the model for graph capture.
# ===----------------------------------------------------------------------===
model = alexnet(weights=None).eval()
model.load_state_dict(
    torch.load(
        model_dir / "../../../models/alexnet/output/alexnet/model.pth",
        weights_only=True,
    )
)
for layer in model.modules():
    if isinstance(layer, torch.nn.ReLU):
        layer.inplace = False


# ===----------------------------------------------------------------------===
# Step 2.1: Capture the original FP32 graph without trace.
# ===----------------------------------------------------------------------===
compiler = DynamoCompiler(primary_registry=tosa.ops_registry, trace=None)
with torch.no_grad():
    (graph,) = compiler.importer(model, torch.zeros((1, 3, 224, 224)))
params = compiler.imported_params[graph]
graph.dump(output / "01-import.txt")
# graph.dump_parameters(output / "01-import-parameters")


# ===----------------------------------------------------------------------===
# Step 2.2: Normalize activation layouts while preserving checkpoint weight axes.
# ===----------------------------------------------------------------------===
names = {p.data_ptr(): name for name, p in model.named_parameters()}
parameter_names = {
    node.name: names[p.data_ptr()] for node, p in zip(graph.params, params, strict=True)
}
permute(graph)
graph.dump(output / "02-permuted.txt")
# graph.dump_parameters(output / "02-permuted-parameters")


# ===----------------------------------------------------------------------===
# Step 2.3: Quantize weights and calibrate fixed activation scales offline.
# Calibration uses ImageNet; export graph and parameter data after permutation.
# ===----------------------------------------------------------------------===
config = tomllib.loads((model_dir / "quant/quant.toml").read_text())
params = quantize(
    graph,
    config,
    parameter_names=parameter_names,
    calibration_model=model,
    calibration_dataset=ImageNetCalibration(),
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
(output / "labels.txt").write_text(
    "\n".join(AlexNet_Weights.IMAGENET1K_V1.meta["categories"]) + "\n"
)


# ===----------------------------------------------------------------------===
# Step 4: Export parameter buffers in ABI order.
# arg0.data contains FP32 parameters and scales; arg1.data contains INT8 weights.
# ===----------------------------------------------------------------------===
for index, dtype in enumerate((torch.float32, torch.int8)):
    np.concatenate(
        [p.detach().cpu().numpy().reshape(-1) for p in params if p.dtype == dtype]
    ).tofile(output / f"arg{index}.data")
