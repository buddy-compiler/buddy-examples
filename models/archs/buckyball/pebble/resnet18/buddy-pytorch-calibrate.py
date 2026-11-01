from importlib import import_module
from pathlib import Path
import json

import torch
from torch.nn.utils.fusion import fuse_conv_bn_eval

from compiler.frontend.quant import quantize, quantize_bias

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
imagenet = import_module("models.models.resnet18.data")


# ===----------------------------------------------------------------------===
# Load the checkpoint and fold every batch normalization into the
# convolution before it, as the importer does.
# ===----------------------------------------------------------------------===
resnet18 = import_module("models.models.resnet18.model")


def load_folded():
    model = resnet18.load()
    for module in model.modules():
        names = list(module._modules)
        for conv, norm in zip(names, names[1:]):
            if isinstance(module._modules[conv], torch.nn.Conv2d) and isinstance(
                module._modules[norm], torch.nn.BatchNorm2d
            ):
                module._modules[conv] = fuse_conv_bn_eval(
                    module._modules[conv], module._modules[norm]
                )
                module._modules[norm] = torch.nn.Identity()
    return model


model = load_folded()
convolutions = {
    name: module
    for name, module in model.named_modules()
    if isinstance(module, torch.nn.Conv2d)
}
images, _ = imagenet.load("train-*.parquet", 12)

# ===----------------------------------------------------------------------===
# Record the largest magnitude of every input channel of every convolution
# and the largest input magnitude of the fc Linear over about 1024 training
# images.
# ===----------------------------------------------------------------------===
channel_maxima = {}
maxima = {}


def record_channels(name, x):
    maximum = x.abs().amax([0, 2, 3])
    channel_maxima[name] = torch.maximum(channel_maxima.get(name, maximum), maximum)


hooks = [
    module.register_forward_pre_hook(
        lambda module, inputs, name=name: record_channels(name, inputs[0])
    )
    for name, module in convolutions.items()
]
hooks.append(
    model.fc.register_forward_pre_hook(
        lambda module, inputs: maxima.update(
            {"fc": max(maxima.get("fc", 0.0), inputs[0].abs().max().item())}
        )
    )
)
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
# magnitudes: a few outliers would otherwise take most of the 256 codes. The
# fc Linear converts its whole input with one multiplier.
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
# are rejected when the accuracy drops more than 0.5 points below FP32.
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
quantized = load_folded()
for name, module in quantized.named_modules():
    if not isinstance(module, torch.nn.Conv2d):
        continue
    scale = torch.tensor(clips[name] / 127)
    weight = module.weight.detach() * smooth[name].reshape(1, -1, 1, 1)
    # A channel whose folded weights almost vanish takes a coarser step, so
    # that its bias code stays within 2^30.
    maximum = weight.abs().amax([1, 2, 3], keepdim=True)
    floor = module.bias.detach().abs().reshape(maximum.shape) / (scale * 2**30)
    step = torch.maximum(maximum / 127, floor)
    module.weight.data = quantize(weight, 1 / step) * step
    bias_step = step.reshape(-1) * scale
    module.bias.data = quantize_bias(module.bias.detach(), bias_step) * bias_step
    module.register_forward_pre_hook(
        lambda module, inputs, name=name, scale=scale: quantize(
            inputs[0] / smooth[name].reshape(1, -1, 1, 1), 1 / scale
        )
        * scale
    )
module = quantized.fc
step = module.weight.detach().abs().amax(1, keepdim=True) / 127
module.weight.data = quantize(module.weight.detach(), 1 / step) * step
scale = torch.tensor(maxima["fc"] / 127, dtype=torch.float32)
bias_step = step.reshape(-1) * scale
module.bias.data = quantize_bias(module.bias.detach(), bias_step) * bias_step
module.register_forward_pre_hook(
    lambda module, inputs: quantize(inputs[0], 1 / scale) * scale
)
int8 = accuracy(quantized)
print(f"FP32 accuracy {fp32:.2f}%, INT8 accuracy {int8:.2f}% on {len(labels)} images")
if fp32 - int8 > 0.5:
    raise RuntimeError(
        f"INT8 accuracy {int8:.2f}% is more than 0.5 below FP32 {fp32:.2f}%"
    )
calibration = {
    name: {"smooth": smooth[name].tolist(), "clip": clips[name]} for name in clips
}
calibration["fc"] = maxima["fc"]
(output / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
