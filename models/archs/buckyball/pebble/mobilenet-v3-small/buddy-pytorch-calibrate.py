from importlib import import_module
from pathlib import Path
import json

import torch
from torch.nn.utils.fusion import fuse_conv_bn_eval

from compiler.frontend.quant import quantize, quantize_bias

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
imagenet = import_module("models.models.mobilenet-v3-small.data")

# ===----------------------------------------------------------------------===
# Load the checkpoint and fold every batch normalization into the
# convolution before it, as the importer does.
# ===----------------------------------------------------------------------===
mobilenet = import_module("models.models.mobilenet-v3-small.model")
model = mobilenet.load()
for block in model.modules():
    if (
        isinstance(block, torch.nn.Sequential)
        and isinstance(block[0], torch.nn.Conv2d)
        and isinstance(block[1], torch.nn.BatchNorm2d)
    ):
        block[0] = fuse_conv_bn_eval(block[0], block[1])
        block[1] = torch.nn.Identity()


# ===----------------------------------------------------------------------===
# Record the largest magnitude of every input channel of every convolution
# over 1024 training images.
# ===----------------------------------------------------------------------===
convolutions = {
    name: module
    for name, module in model.named_modules()
    if isinstance(module, torch.nn.Conv2d)
}
channel_maxima = {}


def record_channels(name, x):
    maximum = x.abs().amax([0, 2, 3])
    channel_maxima[name] = torch.maximum(channel_maxima.get(name, maximum), maximum)


hooks = [
    module.register_forward_pre_hook(
        lambda module, inputs, name=name: record_channels(name, inputs[0])
    )
    for name, module in convolutions.items()
]
with torch.no_grad():
    for batch, _ in imagenet.shuffled("train", 1024):
        model(batch)
for hook in hooks:
    hook.remove()

# ===----------------------------------------------------------------------===
# Smooth the inputs: the input divides every channel by s while it converts
# to INT8 online, the weight takes s. s = sqrt(a / w) from the largest input
# magnitude a and weight magnitude w of the channel; a depthwise convolution
# reads one input channel per output channel, so it takes all of a. Then
# record the largest magnitude of every smoothed input.
# ===----------------------------------------------------------------------===
smooth = {}
for name, module in convolutions.items():
    a = channel_maxima[name].clamp(min=1e-5)
    if module.groups > 1:
        smooth[name] = a
    else:
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
    for batch, _ in imagenet.shuffled("train", 1024):
        model(batch)
for hook in hooks:
    hook.remove()
clips = {
    name: torch.quantile(torch.cat(values), 0.9999).item()
    for name, values in samples.items()
}

# ===----------------------------------------------------------------------===
# Evaluate the quantization in software on 1024 validation images: the
# scales are rejected when the top-1 accuracy drops more than 1 point below
# FP32.
# ===----------------------------------------------------------------------===
tests = list(imagenet.shuffled("val", 1024))


def accuracy(model):
    with torch.no_grad():
        correct = sum(
            (model(batch).argmax(1) == label).sum().item() for batch, label in tests
        )
    return correct / 1024 * 100


fp32 = accuracy(model)
for name, module in convolutions.items():
    # All-zero inputs (an SE excitation): any positive step encodes them.
    scale = torch.tensor(clips[name] / 127 if clips[name] else 1.0)
    shape = (-1, 1, 1, 1) if module.groups > 1 else (1, -1, 1, 1)
    weight = module.weight.detach() * smooth[name].reshape(shape)
    # A channel whose folded weights almost vanish takes a coarser step, so
    # that its bias code stays within 2^30.
    maximum = weight.abs().amax([1, 2, 3], keepdim=True)
    floor = module.bias.detach().abs().reshape(maximum.shape) / (scale * 2**30)
    step = torch.maximum(maximum / 127, floor)
    step = torch.where(step == 0, 1.0, step)
    module.weight.data = quantize(weight, 1 / step) * step
    bias_step = step.reshape(-1) * scale
    module.bias.data = quantize_bias(module.bias.detach(), bias_step) * bias_step
    module.register_forward_pre_hook(
        lambda module, inputs, name=name, scale=scale: quantize(
            inputs[0] / smooth[name].reshape(1, -1, 1, 1), 1 / scale
        )
        * scale
    )
quantized = accuracy(model)
print(f"FP32 top-1 {fp32:.2f}%, INT8 top-1 {quantized:.2f}%")
if fp32 - quantized > 1:
    raise RuntimeError(
        f"INT8 top-1 {quantized:.2f}% is more than 1 below FP32 {fp32:.2f}%"
    )
calibration = {
    name: {"smooth": smooth[name].tolist(), "clip": clips[name]} for name in clips
}
(output / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
