from importlib import import_module
from pathlib import Path

import torch
from torch.overrides import TorchFunctionMode

from compiler.frontend.quant import dequantize_mxint8, quantize_mxint8

model_dir = Path(__file__).resolve().parent
emotion = import_module("models.models.bert-base-uncased-emotion.data")
bert = import_module("models.models.bert-base-uncased-emotion.model")


def mxint8(values, axis):
    """Values through MXINT8 along `axis`, as the graph computes them."""
    return dequantize_mxint8(quantize_mxint8(values, axis), axis)


class Products(TorchFunctionMode):
    """Reads both operands of every attention product through MXINT8 along
    its reduction axis."""

    def __torch_function__(self, function, types, args=(), kwargs=None):
        # `a @ b` arrives as the matmul method of the tensor.
        if getattr(function, "__name__", None) == "matmul":
            args = (mxint8(args[0], -1), mxint8(args[1], -2))
        return function(*args, **(kwargs or {}))


# ===----------------------------------------------------------------------===
# The 2000 validation sentences as the inputs of the graph, 32 tokens each.
# ===----------------------------------------------------------------------===
tests, labels = emotion.load("validation", 2000)


def accuracy(model, products):
    correct = 0
    with torch.no_grad():
        for batch, label in zip(
            zip(*(value.split(100) for value in tests)), labels.split(100)
        ):
            if products:
                with Products():
                    logits = model(*batch)
            else:
                logits = model(*batch)
            correct += (logits.argmax(1) == label).sum().item()
    return correct / len(labels) * 100


# ===----------------------------------------------------------------------===
# Evaluate MXINT8 in software: the weights of every Linear packed offline, its
# input and both operands of every attention product quantized online. The
# quantization is rejected when the accuracy drops more than 1 point below
# FP32.
# ===----------------------------------------------------------------------===
model = bert.load()
fp32 = accuracy(model, False)
for module in model.modules():
    if isinstance(module, torch.nn.Linear):
        module.weight = torch.nn.Parameter(mxint8(module.weight.detach(), 1))
        module.register_forward_pre_hook(
            lambda module, inputs: (mxint8(inputs[0], -1),)
        )
quantized = accuracy(model, True)
print(f"FP32 accuracy {fp32:.2f}%, MXINT8 accuracy {quantized:.2f}%")
if fp32 - quantized > 1:
    raise RuntimeError(
        f"MXINT8 accuracy {quantized:.2f}% is more than 1 below FP32 {fp32:.2f}%"
    )
