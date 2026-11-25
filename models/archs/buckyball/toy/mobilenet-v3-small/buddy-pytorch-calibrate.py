# ===- buddy-pytorch-calibrate.py ------------------------------------------
#
# MobileNetV3-Small on toy: the largest input magnitude of every convolution
# and Linear (batch normalizations folded) over about 1000 training images,
# in output/calibration.json. The scales are rejected when the INT8 accuracy
# on the validation images of data.load drops more than 1 point below FP32.
#
# ===---------------------------------------------------------------------===

from importlib import import_module
from pathlib import Path
import json
import sys

import torch
from quantized_model import folded, reference

output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=True)
mobilenet = import_module("models.models.mobilenet-v3-small.model")
imagenet = import_module("models.models.mobilenet-v3-small.data")


torch.set_num_threads(4)
model = folded()
images, _ = imagenet.load("train-*.parquet", 12)
# Tile dense pointwise projections and classifier GEMMs on VecBall. Depthwise
# filters and squeeze/excitation gates retain their floating-point computation.
samples = {}
hooks = []
for name, module in model.named_modules():
    pointwise = (
        isinstance(module, torch.nn.Conv2d)
        and module.groups == 1
        and module.kernel_size == (1, 1)
    )
    if not (
        isinstance(module, torch.nn.Linear)
        or (pointwise and not name.endswith((".fc1", ".fc2")))
    ):
        continue
    samples[name] = []
    hooks.append(
        module.register_forward_pre_hook(
            lambda layer, inputs, name=name: samples[name].append(
                inputs[0].detach().flatten()[::257].abs().clone()
            )
        )
    )
with torch.no_grad():
    for batch in images.split(64):
        model(batch)
for hook in hooks:
    hook.remove()
maxima = {
    name: torch.quantile(torch.cat(parts), 0.9999).item()
    for name, parts in samples.items()
}
(output / "training-ranges.json").write_text(json.dumps(maxima, indent=2) + "\n")

tests, labels = imagenet.load("val-*.parquet", 1)


def accuracy(model):
    with torch.no_grad():
        correct = sum(
            (model(batch).argmax(1) == label).sum().item()
            for batch, label in zip(tests.split(100), labels.split(100))
        )
    return correct / len(labels) * 100


fp32 = accuracy(model)
quantized = reference(maxima)
int8 = accuracy(quantized)
print(f"FP32 accuracy {fp32:.2f}%, INT8 accuracy {int8:.2f}% on {len(labels)} images")
if fp32 - int8 > 1:
    raise RuntimeError(
        f"INT8 accuracy {int8:.2f}% is more than 1 point below FP32 {fp32:.2f}%"
    )
(output / "calibration.json").write_text(json.dumps(maxima, indent=2) + "\n")
