"""A slice of ResNet18 small enough for an RTL simulation of the Gemmini
device: the last residual block and the classifier (layer4[1], avgpool, fc)
of the model, compiled as the whole model is (buddy-pytorch-importer.py) with
its calibration (calibration.json). Its input is the output of layer4[0] for
the
first check image and its output the model's logits for it, both computed
the way the compiled graph computes them, as buddy-pytorch-calibrate.py
computes the logits. (The first layers do not fit an RTL simulation: Gemmini
has no conversion hardware, so the CPU quantizes and normalizes every
activation, 800 k values after conv1 alone.)"""

from importlib import import_module
from pathlib import Path
import json

import numpy as np
import torch

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from compiler.frontend import quant
from compiler.frontend.quant import quantize as quantize_value, quantize_bias
from frontend.permute.graph import permute
from frontend.quant.graph import quantize

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
sliced = output / "slice"
sliced.mkdir(parents=True, exist_ok=True)
resnet18 = import_module("models.models.resnet18.model")


class Slice(torch.nn.Module):
    """The last modules of ResNet18 under their names in the model, so the
    calibration and the parameters keep their keys."""

    def __init__(self, model):
        super().__init__()
        # A slice of a Sequential keeps its keys: layer4.1.conv1, ...
        self.layer4 = model.layer4[1:]
        self.avgpool, self.fc = model.avgpool, model.fc

    def forward(self, x):
        return self.fc(torch.flatten(self.avgpool(self.layer4(x)), 1))


# ===----------------------------------------------------------------------===
# The slice compiled as buddy-pytorch-importer.py compiles the model.
# ===----------------------------------------------------------------------===
model = Slice(resnet18.load())
compiler = DynamoCompiler(primary_registry={**tosa.ops_registry, **quant.ref_registry})
with torch.no_grad():
    (graph,) = compiler.importer(model, torch.zeros((1, 512, 7, 7)))
params = compiler.imported_params[graph]
names = {
    value.data_ptr(): name
    for name, value in [*model.named_parameters(), *model.named_buffers()]
}
parameter_names = {
    node.name: names[value.data_ptr()]
    for node, value in zip(graph.params, params, strict=True)
}
permute(graph)
calibration = json.loads((output / "calibration.json").read_text())
params = quantize(graph, params, parameter_names, calibration)
graph.fuse_ops([simply_fuse])
driver = GraphDriver(graph)
for index, subgraph in enumerate(driver.subgraphs):
    subgraph.lower_to_top_level_ir()
    (sliced / f"subgraph{index}.mlir").write_text(str(subgraph._imported_module))
(sliced / "forward.mlir").write_text(str(driver.construct_main_graph(True)))
# The convolutions of the slice have no bias: its INT32 group is empty.
for index, dtype in enumerate((torch.float32, torch.int32, torch.int8)):
    group = [p.detach().numpy().reshape(-1) for p in params if p.dtype == dtype]
    np.concatenate(group or [np.zeros(0, np.int32)]).tofile(sliced / f"arg{index}.data")


# ===----------------------------------------------------------------------===
# The input and output of the slice for the first check image: the reference
# model of buddy-pytorch-calibrate.py, operation for operation as the
# compiled graph computes it, its layer4[0] output and its logits.
# ===----------------------------------------------------------------------===
def mean(values, dims):
    total = values
    count = 1
    for d in dims:
        moved = total.movedim(d, 0)
        accumulated = torch.zeros_like(moved[0])
        for index in range(moved.shape[0]):
            accumulated = accumulated + moved[index]
        total = accumulated.unsqueeze(d)
        count *= values.shape[d]
    return total * torch.tensor(1.0 / count, dtype=torch.float32)


def inputs(x):
    return [1, -1, 1, 1] if x.ndim == 4 else [1] * (x.ndim - 1) + [-1]


smooth = {name: torch.tensor(c["smooth"]) for name, c in calibration.items()}
maxima = {name: c["maximum"] for name, c in calibration.items()}
reference = resnet18.load()
for name, module in reference.named_modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.register_forward_hook(
            lambda module, args, output: (
                (args[0] - module.running_mean.reshape(1, -1, 1, 1))
                * (1 / torch.sqrt(module.running_var + module.eps)).reshape(1, -1, 1, 1)
            )
            * module.weight.reshape(1, -1, 1, 1)
            + module.bias.reshape(1, -1, 1, 1)
        )
    if isinstance(module, torch.nn.AdaptiveAvgPool2d):
        module.register_forward_hook(lambda module, args, output: mean(args[0], [2, 3]))
    if not isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
        continue
    weight = module.weight.detach() * smooth[name].reshape(
        [1, -1] + [1] * (module.weight.ndim - 2)
    )
    maximum = weight.abs().amax(list(range(1, weight.ndim)), keepdim=True)
    step = torch.where(maximum == 0, 1.0, maximum / 127)
    scale = maxima[name] / 127 if maxima[name] else 1.0
    accumulator_step = (step * scale).reshape(-1)
    if module.bias is not None:
        module.bias.data = quantize_bias(
            module.bias.detach(), accumulator_step
        ).double()
    module.weight.data = quantize_value(weight, 1 / step).double()
    module.register_forward_pre_hook(
        lambda module, args, name=name, scale=scale: quantize_value(
            args[0], 1 / (scale * smooth[name].reshape(inputs(args[0])))
        ).double()
    )
    module.register_forward_hook(
        lambda module, args, output, accumulator_step=accumulator_step: output.float()
        * accumulator_step.reshape([1, -1, 1, 1] if output.ndim == 4 else [1, -1])
    )
stage = {}
reference.layer4[0].register_forward_hook(
    lambda module, args, output: stage.setdefault("input", output)
)
image = torch.from_numpy(
    np.fromfile(output / "check-images.data", dtype=np.float32, count=3 * 224 * 224)
).reshape(1, 3, 224, 224)
with torch.no_grad():
    expected = reference(image)
stage["input"].numpy().astype("float32").tofile(sliced / "check-input.data")
expected.numpy().astype("float32").tofile(sliced / "check-output.data")
logits = np.fromfile(output / "check-logits.data", dtype=np.float32, count=1000)
if not np.array_equal(logits, expected.numpy().reshape(-1)):
    raise RuntimeError("the reference differs from the model's check logits")
print(
    f"resnet18 slice: input {list(stage['input'].shape)}, output {list(expected.shape)}"
)
