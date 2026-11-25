from importlib import import_module
from pathlib import Path
import json

import torch

torch.set_num_threads(8)
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
# Record the largest magnitude of every input channel of every convolution and Linear
# over 1024 training images.
# ===----------------------------------------------------------------------===
layers = {
    name: module
    for name, module in model.named_modules()
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
}
channel_maxima = {}


def record_channels(name, x):
    maximum = x.abs().amax([0, 2, 3] if x.ndim == 4 else [0])
    channel_maxima[name] = torch.maximum(channel_maxima.get(name, maximum), maximum)


hooks = [
    module.register_forward_pre_hook(
        lambda module, inputs, name=name: record_channels(name, inputs[0])
    )
    for name, module in layers.items()
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
for name, module in layers.items():
    a = channel_maxima[name].clamp(min=1e-5)
    if isinstance(module, torch.nn.Linear):
        smooth[name] = torch.ones(1, device=a.device)
    elif module.groups > 1:
        smooth[name] = a
    else:
        w = module.weight.detach().abs().amax([0, 2, 3]).clamp(min=1e-5)
        smooth[name] = (a / w).sqrt()
torch.manual_seed(0)
samples = {}


def sample(name, x):
    x = (x / smooth[name].reshape([1, -1] + [1] * (x.ndim - 2))).abs().flatten()
    samples.setdefault(name, []).append(
        x[torch.randint(x.numel(), (min(200000, x.numel()),), device=x.device)]
    )


hooks = [
    module.register_forward_pre_hook(
        lambda module, inputs, name=name: sample(name, inputs[0])
    )
    for name, module in layers.items()
]
with torch.no_grad():
    for batch, _ in imagenet.shuffled("train", 1024):
        model(batch)
for hook in hooks:
    hook.remove()
clips = {
    name: (
        channel_maxima[name].max().item()
        if isinstance(layers[name], torch.nn.Linear)
        else torch.quantile(torch.cat(values), 0.9999).item()
    )
    for name, values in samples.items()
}
calibration = {
    name: {"smooth": smooth[name].tolist(), "clip": clips[name]} for name in clips
}
(output / "calibration-ranges.json").write_text(
    json.dumps(calibration, indent=2) + "\n"
)

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
from frontend.fixed.reference import FixedGraph

fixed = FixedGraph(model, calibration)
quantized = accuracy(fixed.run)
print(f"FP32 top-1 {fp32:.2f}%, INT8 top-1 {quantized:.2f}%")
if fp32 - quantized > 1:
    raise RuntimeError(
        f"INT8 top-1 {quantized:.2f}% is more than 1 below FP32 {fp32:.2f}%"
    )
calibration = {
    name: {"smooth": smooth[name].tolist(), "clip": clips[name]} for name in clips
}
calibration["validation"] = {
    "samples": 1024,
    "fp32_top1": fp32,
    "int8_top1": quantized,
    "fraction_bits": 24,
    "clipped_values": fixed.clipped,
}
(output / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
