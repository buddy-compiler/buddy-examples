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
        # `a @ b` arrives as the matmul method of the tensor.
        if getattr(function, "__name__", None) == "matmul":
            args = self.hook(self.index, args)
            self.index += 1
        return function(*args, **(kwargs or {}))


# ===----------------------------------------------------------------------===
# Record the largest magnitude of the input of every Linear and of both
# operands of every attention product over 1024 training sentences.
# ===----------------------------------------------------------------------===
model = bert.load()
inputs, _ = emotion.load("train", 1024)
maxima = {}


def record(name, value):
    maxima[name] = max(maxima.get(name, 0.0), value.abs().max().item())


def record_product(index, operands):
    record(f"matmul.{index}.0", operands[0])
    record(f"matmul.{index}.1", operands[1])
    return operands


hooks = [
    module.register_forward_pre_hook(
        lambda module, args, name=name: record(name, args[0])
    )
    for name, module in model.named_modules()
    if isinstance(module, torch.nn.Linear)
]
with torch.no_grad():
    for batch in zip(*(value.split(64) for value in inputs)):
        with Products(record_product):
            model(*batch)
for hook in hooks:
    hook.remove()

# ===----------------------------------------------------------------------===
# Evaluate the quantization in software on the 2000 validation sentences: the
# scales are rejected when the accuracy drops more than 1 point below FP32.
# Every Linear reads its weight with one scale per output channel and its
# input with one scale; every attention product reads both operands with one
# scale each.
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
    scales = [torch.tensor(maxima[f"matmul.{index}.{side}"] / 127) for side in (0, 1)]
    return tuple(
        quantize(operand, 1 / scale) * scale for operand, scale in zip(operands, scales)
    )


fp32 = accuracy(model)
quantized = bert.load()
for name, module in quantized.named_modules():
    if not isinstance(module, torch.nn.Linear):
        continue
    step = module.weight.detach().abs().amax(1, keepdim=True) / 127
    module.weight.data = quantize(module.weight.detach(), 1 / step) * step
    scale = torch.tensor(maxima[name] / 127)
    bias_step = step.reshape(-1) * scale
    module.bias.data = quantize_bias(module.bias.detach(), bias_step) * bias_step
    module.register_forward_pre_hook(
        lambda module, args, scale=scale: quantize(args[0], 1 / scale) * scale
    )
int8 = accuracy(quantized, fake_product)
print(
    f"FP32 accuracy {fp32:.2f}%, INT8 accuracy {int8:.2f}% on {len(labels)} sentences"
)
if fp32 - int8 > 1:
    raise RuntimeError(
        f"INT8 accuracy {int8:.2f}% is more than 1 below FP32 {fp32:.2f}%"
    )
(output / "calibration.json").write_text(json.dumps(maxima, indent=2) + "\n")
