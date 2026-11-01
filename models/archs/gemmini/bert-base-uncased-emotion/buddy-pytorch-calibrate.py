from importlib import import_module
from pathlib import Path
import json

import torch
from torch.overrides import TorchFunctionMode

from compiler.frontend.quant import quantize, quantize_bias

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
emotion = import_module("models.models.bert-base-uncased-emotion.data")
bert = import_module("models.models.bert-base-uncased-emotion.model")


class Products(TorchFunctionMode):
    """Calls `hook(index, operands)` on every attention product, numbered in
    the order the encoder runs them, and computes the product of what the
    hook returns."""

    def __init__(self, hook):
        super().__init__()
        self.hook = hook
        self.index = 0

    def __torch_function__(self, function, types, args=(), kwargs=None):
        if function is torch.matmul:
            args = self.hook(self.index, args)
            self.index += 1
        return function(*args, **(kwargs or {}))


# ===----------------------------------------------------------------------===
# Record the largest magnitude of every input channel of every Linear and of
# both operands of every attention product over 1024 training sentences.
# ===----------------------------------------------------------------------===
model = bert.load()
inputs, _ = emotion.load("train", 1024)
batches = list(zip(*(value.split(64) for value in inputs)))
channel_maxima = {}
maxima = {}


def record(name, value):
    maxima[name] = max(maxima.get(name, 0.0), value.abs().max().item())


def record_channels(name, x):
    maximum = x.abs().amax(list(range(x.ndim - 1)))
    channel_maxima[name] = torch.maximum(channel_maxima.get(name, maximum), maximum)


def record_product(index, operands):
    record(f"matmul.{index}.0", operands[0])
    record(f"matmul.{index}.1", operands[1])
    return operands


hooks = [
    module.register_forward_pre_hook(
        lambda module, args, name=name: record_channels(name, args[0])
    )
    for name, module in model.named_modules()
    if isinstance(module, torch.nn.Linear)
]
with torch.no_grad():
    for batch in batches:
        with Products(record_product):
            model(*batch)
for hook in hooks:
    hook.remove()

# ===----------------------------------------------------------------------===
# Smooth the Linear inputs: the CPU divides every input channel by s while it
# converts the input to INT8 online, the weight takes s. s = a^0.75 / w^0.25
# from the largest input magnitude a and weight magnitude w of the channel.
# Then record the largest magnitude of every smoothed input.
# ===----------------------------------------------------------------------===
smooth = {}
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        a = channel_maxima[name].clamp(min=1e-5)
        w = module.weight.detach().abs().amax(0).clamp(min=1e-5)
        smooth[name] = a**0.75 / w**0.25


def record_smoothed(name, x):
    record(name, x / smooth[name])


hooks = [
    module.register_forward_pre_hook(
        lambda module, args, name=name: record_smoothed(name, args[0])
    )
    for name, module in model.named_modules()
    if isinstance(module, torch.nn.Linear)
]
with torch.no_grad():
    for batch in batches:
        model(*batch)
for hook in hooks:
    hook.remove()

# ===----------------------------------------------------------------------===
# Evaluate the quantization in software on the 2000 validation sentences: the
# scales are rejected when the accuracy drops more than 1 point below FP32.
# ===----------------------------------------------------------------------===
tests, labels = emotion.load("validation", 2000)


def accuracy(model, products=None):
    correct = 0
    with torch.no_grad():
        for batch, label in zip(
            zip(*(value.split(100) for value in tests)), labels.split(100)
        ):
            if products is None:
                logits = model(*batch)
            else:
                with Products(products):
                    logits = model(*batch)
            correct += (logits.argmax(1) == label).sum().item()
    return correct / len(labels) * 100


def fake_product(index, operands):
    return tuple(
        quantize(operand, 127 / maxima[f"matmul.{index}.{side}"])
        * (maxima[f"matmul.{index}.{side}"] / 127)
        for side, operand in enumerate(operands)
    )


fp32 = accuracy(model)
quantized = bert.load()
for name, module in quantized.named_modules():
    if not isinstance(module, torch.nn.Linear):
        continue
    weight = module.weight.detach() * smooth[name]
    step = weight.abs().amax(1, keepdim=True) / 127
    module.weight.data = quantize(weight, 1 / step) * step
    scale = torch.tensor(maxima[name] / 127)
    bias_step = step.reshape(-1) * scale
    module.bias.data = quantize_bias(module.bias.detach(), bias_step) * bias_step
    module.register_forward_pre_hook(
        lambda module, args, name=name, scale=scale: quantize(
            args[0] / smooth[name], 1 / scale
        )
        * scale
    )
quantized_accuracy = accuracy(quantized, fake_product)
print(f"FP32 accuracy {fp32:.2f}%, INT8 accuracy {quantized_accuracy:.2f}%")
if fp32 - quantized_accuracy > 1:
    raise RuntimeError(
        f"INT8 accuracy {quantized_accuracy:.2f}% is more than 1 below FP32 {fp32:.2f}%"
    )
calibration = {
    name: (
        {"smooth": smooth[name].tolist(), "maximum": maxima[name]}
        if name in smooth
        else maxima[name]
    )
    for name in maxima
}
(output / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
