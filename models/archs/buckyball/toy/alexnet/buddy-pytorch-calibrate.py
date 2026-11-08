from importlib import import_module
from pathlib import Path
import json

import torch

from compiler.frontend.quant import quantize, quantize_bias
from frontend.quant.graph import smooth_classifier

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
imagenet = import_module("models.models.alexnet.data")


alexnet = import_module("models.models.alexnet.model")
model = alexnet.load()
images, _ = imagenet.load("train-*.parquet", 12)

# ===----------------------------------------------------------------------===
# Smooth the classifier Linears through their producers: s = sqrt(a / w)
# from the largest input magnitude a and weight magnitude w of every input
# channel of a Linear; classifier.1 takes one factor per channel of
# features.10, over its 6x6 positions.
# ===----------------------------------------------------------------------===
input_maxima = {}


def record_inputs(name, x):
    maximum = x.abs().amax(0)
    input_maxima[name] = torch.maximum(input_maxima.get(name, maximum), maximum)


hooks = [
    model.classifier[index].register_forward_pre_hook(
        lambda module, inputs, name=f"classifier.{index}": record_inputs(
            name, inputs[0]
        )
    )
    for index in (1, 4, 6)
]
with torch.no_grad():
    for batch in images.split(64):
        model(batch)
for hook in hooks:
    hook.remove()
classifier_smoothing = {}
for index in (1, 4, 6):
    name = f"classifier.{index}"
    positions = 36 if index == 1 else 1
    a = input_maxima[name].reshape(-1, positions).amax(1).clamp(min=1e-5)
    w = model.classifier[index].weight.detach().abs().amax(0)
    w = w.reshape(-1, positions).amax(1).clamp(min=1e-5)
    classifier_smoothing[name] = (a / w).sqrt().tolist()
smooth_classifier(model, classifier_smoothing)
convolutions = {
    name: module
    for name, module in model.named_modules()
    if isinstance(module, torch.nn.Conv2d)
}
linears = {
    name: module
    for name, module in model.named_modules()
    if isinstance(module, torch.nn.Linear)
}

# ===----------------------------------------------------------------------===
# Record the largest magnitude of every input channel of every convolution
# and the largest input magnitude of every Linear over about 1024 training
# images.
# ===----------------------------------------------------------------------===
channel_maxima = {}
maxima = {}


def record_channels(name, x):
    maximum = x.abs().amax([0, 2, 3])
    channel_maxima[name] = torch.maximum(channel_maxima.get(name, maximum), maximum)


def record(name, x):
    maxima[name] = max(maxima.get(name, 0.0), x.abs().max().item())


hooks = [
    module.register_forward_pre_hook(
        lambda module, inputs, name=name: record_channels(name, inputs[0])
    )
    for name, module in convolutions.items()
] + [
    module.register_forward_pre_hook(
        lambda module, inputs, name=name: record(name, inputs[0])
    )
    for name, module in linears.items()
]
with torch.no_grad():
    for batch in images.split(64):
        model(batch)
for hook in hooks:
    hook.remove()

# ===----------------------------------------------------------------------===
# Smooth the convolution inputs: the input divides every channel by s while
# it converts to INT8 online, the weight takes s. s = sqrt(a / w) from the
# largest input magnitude a and weight magnitude w of the channel. Then
# sample every smoothed input and clip it at the 99.99th percentile of its
# magnitudes: a few outliers would otherwise take most of the 256 codes. A
# Linear converts its whole input with one multiplier.
# ===----------------------------------------------------------------------===
smooth = {}
for name, module in convolutions.items():
    a = channel_maxima[name].clamp(min=1e-5)
    w = module.weight.detach().abs().amax([0, 2, 3]).clamp(min=1e-5)
    smooth[name] = (a / w).sqrt()

torch.manual_seed(0)
samples = {}


def sample(name, x):
    x = (x / smooth[name].reshape(1, -1, 1, 1)).abs().flatten()
    samples.setdefault(name, []).append(x[torch.randperm(x.numel())[:200000]])


hooks = [
    module.register_forward_pre_hook(
        lambda module, inputs, name=name: sample(name, inputs[0])
    )
    for name, module in convolutions.items()
]
with torch.no_grad():
    for batch in images.split(64):
        model(batch)
for hook in hooks:
    hook.remove()
clips = {
    name: torch.quantile(torch.cat(values), 0.9999).item()
    for name, values in samples.items()
}

# ===----------------------------------------------------------------------===
# Evaluate the quantization in software on 500 validation images: the scales
# are rejected when the accuracy drops more than 1 point below FP32.
# ===----------------------------------------------------------------------===
tests, labels = imagenet.load("val-*.parquet", 1)


def accuracy(model):
    with torch.no_grad():
        correct = sum(
            (model(batch).argmax(1) == label).sum().item()
            for batch, label in zip(tests.split(100), labels.split(100))
        )
    return correct / len(labels) * 100


fp32 = accuracy(model)
quantized = alexnet.load()
smooth_classifier(quantized, classifier_smoothing)
for name, module in quantized.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        scale = torch.tensor(clips[name] / 127)
        weight = module.weight.detach() * smooth[name].reshape(1, -1, 1, 1)
        step = weight.abs().amax([1, 2, 3], keepdim=True) / 127
        module.weight.data = quantize(weight, 1 / step) * step
        module.register_forward_pre_hook(
            lambda module, inputs, name=name, scale=scale: quantize(
                inputs[0] / smooth[name].reshape(1, -1, 1, 1), 1 / scale
            )
            * scale
        )
    elif isinstance(module, torch.nn.Linear):
        scale = torch.tensor(maxima[name] / 127)
        step = module.weight.detach().abs().amax(1, keepdim=True) / 127
        module.weight.data = quantize(module.weight.detach(), 1 / step) * step
        module.register_forward_pre_hook(
            lambda module, inputs, scale=scale: quantize(inputs[0], 1 / scale) * scale
        )
    else:
        continue
    bias_step = step.reshape(-1) * scale
    module.bias.data = quantize_bias(module.bias.detach(), bias_step) * bias_step
int8 = accuracy(quantized)
print(f"FP32 accuracy {fp32:.2f}%, INT8 accuracy {int8:.2f}% on {len(labels)} images")
if fp32 - int8 > 1:
    raise RuntimeError(
        f"INT8 accuracy {int8:.2f}% is more than 1 point below FP32 {fp32:.2f}%"
    )
calibration = {
    name: {"smooth": smooth[name].tolist(), "clip": clips[name]} for name in clips
}
calibration.update(maxima)
calibration["smooth_classifier"] = classifier_smoothing
(output / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
