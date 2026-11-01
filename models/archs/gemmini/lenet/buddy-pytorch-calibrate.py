from importlib import import_module
from pathlib import Path
import json

import torch
from torchvision import datasets, transforms

from compiler.frontend.quant import quantize, quantize_bias

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
mnist = model_dir.parents[2] / "models/lenet/output/datasets"
preprocess = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)


def load(train, count):
    """The first `count` images of the MNIST split and their labels."""
    data = datasets.MNIST(mnist, train=train, download=False, transform=preprocess)
    images, labels = zip(*(data[index] for index in range(count)))
    return torch.stack(images), torch.tensor(labels)


def int8(module):
    """Gemmini computes in INT8 only: every convolution and Linear."""
    return isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))


def depthwise(module):
    return isinstance(module, torch.nn.Conv2d) and module.groups > 1


def channels(module):
    """The broadcast shape of one value per input channel of the weight:
    axis 0 of the [C, 1, H, W] depthwise weights, axis 1 of the others."""
    if depthwise(module):
        return [-1, 1, 1, 1]
    return [1, -1] + [1] * (module.weight.ndim - 2)


def inputs(x):
    """The broadcast shape of one value per channel of an input: axis 1 of a
    convolution input, the last axis of a Linear one."""
    return [1, -1, 1, 1] if x.ndim == 4 else [1] * (x.ndim - 1) + [-1]


# ===----------------------------------------------------------------------===
# Record the largest magnitude of every input channel of every convolution
# and Linear of the checkpoint over 1024 training images.
# ===----------------------------------------------------------------------===
lenet = import_module("models.models.lenet.model")
model = lenet.load()
images, _ = load(True, 1024)
channel_maxima = {}


def record_channels(name, x):
    axes = [
        axis for axis in range(x.ndim) if axis != (1 if x.ndim == 4 else x.ndim - 1)
    ]
    maximum = x.abs().amax(axes)
    channel_maxima[name] = torch.maximum(channel_maxima.get(name, maximum), maximum)


hooks = [
    module.register_forward_pre_hook(
        lambda module, args, name=name: record_channels(name, args[0])
    )
    for name, module in model.named_modules()
    if int8(module)
]
with torch.no_grad():
    for batch in images.split(64):
        model(batch)
for hook in hooks:
    hook.remove()

# ===----------------------------------------------------------------------===
# Smooth the inputs: the CPU divides every input channel by s while it
# converts the input to INT8 online, the weight takes s. s = a^0.75 / w^0.25
# from the largest input magnitude a and weight magnitude w of the channel; a
# depthwise convolution reads one input channel per output channel, so it
# takes all of a. Then record the largest magnitude of every smoothed input.
# ===----------------------------------------------------------------------===
smooth = {}
for name, module in model.named_modules():
    if not int8(module):
        continue
    a = channel_maxima[name].clamp(min=1e-5)
    if depthwise(module):
        smooth[name] = a
    else:
        weight = module.weight.detach().abs()
        w = weight.amax([axis for axis in range(weight.ndim) if axis != 1])
        smooth[name] = a**0.75 / w.clamp(min=1e-5) ** 0.25
maxima = {}


def record_smoothed(name, x):
    x = x / smooth[name].reshape(inputs(x))
    maxima[name] = max(maxima.get(name, 0.0), x.abs().max().item())


hooks = [
    module.register_forward_pre_hook(
        lambda module, args, name=name: record_smoothed(name, args[0])
    )
    for name, module in model.named_modules()
    if int8(module)
]
with torch.no_grad():
    for batch in images.split(64):
        model(batch)
for hook in hooks:
    hook.remove()

# ===----------------------------------------------------------------------===
# Evaluate the quantization in software on the 10000 test images: the scales
# are rejected when the accuracy drops more than 0.5 points below FP32.
# ===----------------------------------------------------------------------===
tests, labels = load(False, 10000)


def accuracy(model):
    with torch.no_grad():
        correct = sum(
            (model(batch).argmax(1) == label).sum().item()
            for batch, label in zip(tests.split(100), labels.split(100))
        )
    return correct / len(labels) * 100


fp32 = accuracy(model)
quantized = lenet.load()
for name, module in quantized.named_modules():
    if not int8(module):
        continue
    weight = module.weight.detach() * smooth[name].reshape(channels(module))
    axes = list(range(1, weight.ndim))
    maximum = weight.abs().amax(axes, keepdim=True)
    # A padding channel is zero; any positive step encodes it.
    step = torch.where(maximum == 0, 1.0, maximum / 127)
    module.weight.data = quantize(weight, 1 / step) * step
    # All-zero inputs (an SE excitation): any positive step encodes them.
    scale = torch.tensor(maxima[name] / 127 if maxima[name] else 1.0)
    if module.bias is not None:
        bias_step = step.reshape(-1) * scale
        module.bias.data = quantize_bias(module.bias.detach(), bias_step) * bias_step
    module.register_forward_pre_hook(
        lambda module, args, name=name, scale=scale: quantize(
            args[0] / smooth[name].reshape(inputs(args[0])), 1 / scale
        )
        * scale
    )
quantized_accuracy = accuracy(quantized)
print(
    f"FP32 accuracy {fp32:.2f}%, INT8 accuracy {quantized_accuracy:.2f}% "
    f"on {len(labels)} images"
)
if fp32 - quantized_accuracy > 0.5:
    raise RuntimeError(
        f"INT8 accuracy {quantized_accuracy:.2f}% is more than 0.5 below FP32 {fp32:.2f}%"
    )
calibration = {
    name: {"smooth": smooth[name].tolist(), "maximum": maxima[name]} for name in maxima
}
(output / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
