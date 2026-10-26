import json
from pathlib import Path

import numpy as np
import torch
from torch._decomp import get_decompositions


from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from stack.compiler.frontend.ops import tosa
from buddy.compiler.trace import TraceConfig, load_trace_config

output = Path("output").resolve()
output.mkdir(parents=True, exist_ok=True)
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

model = mobilenet_v3_small(weights=None)
model.load_state_dict(
    torch.load(
        (
            Path(__file__).resolve().parent
            / "../../../../models/mobilenet-v3-small/output/mobilenet-v3-small/model.pth"
        ).resolve(),
        weights_only=True,
    )
)
for layer in model.modules():
    if isinstance(layer, (torch.nn.ReLU, torch.nn.Hardswish)):
        layer.inplace = False
data = torch.zeros((1, 3, 224, 224))
model.eval()
compiler = DynamoCompiler(
    aot_autograd_decomposition=get_decompositions(
        [torch.ops.aten.hardsigmoid.default, torch.ops.aten.hardswish.default]
    ),
    primary_registry=tosa.ops_registry,
    trace=TraceConfig(load_trace_config(Path(__file__).parent / "trace/trace.toml")),
)
with torch.no_grad():
    graphs = compiler.importer(model, data)
assert len(graphs) == 1
graph = graphs[0]
params = compiler.imported_params[graph]
graph.fuse_ops([simply_fuse])
driver = GraphDriver(graph)
for index, subgraph in enumerate(driver.subgraphs):
    subgraph.lower_to_top_level_ir()
    (output / f"subgraph{index}.mlir").write_text(str(subgraph._imported_module))
(output / "forward.mlir").write_text(str(driver.construct_main_graph(True)))
parameters = np.concatenate(
    [parameter.detach().cpu().numpy().reshape(-1) for parameter in params]
)
parameters.astype(np.float32).tofile(output / "arg0.data")
