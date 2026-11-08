from importlib import import_module
from pathlib import Path
import math

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from compiler.frontend.quant import dequantize_mxint8, quantize_mxint8

model_dir = Path(__file__).resolve().parent
decisions = import_module("models.models.laya.data")
laya = import_module("models.models.laya.model")
# The recipe runs the checkpoint fine-tuned on typed decisions.
laya.checkpoint = laya.checkpoint / "typed-decisions"


def mxint8(values, axis):
    """Values through MXINT8 along `axis`, as the graph computes them."""
    return dequantize_mxint8(quantize_mxint8(values, axis), axis)


def split(values, axis):
    """Values as the two MXINT8 terms the graph reads the key with."""
    head = mxint8(values, axis)
    return head + mxint8(values - head, axis)


class Products(TorchFunctionMode):
    """Reads both operands of every matrix product through MXINT8 along its
    reduction axis K, the key of the score product q k^T as two MXINT8 terms;
    a product whose K is not a multiple of 32 stays FP32: the first Linear of
    the act head, and the outer product of the rotary angles (a broadcast
    multiply in the graph). Every Linear reads its weight [N, K] packed on K
    and its input. The attention of the head runs as the imported graph runs
    it: both operands of the scores take the square root of the softmax
    scale. It reaches the mode as one multi_head_attention_forward call: its
    body runs under the mode, without dispatching to it again, so that its
    Linears and products reach it."""

    probabilities = False

    def __torch_function__(self, function, types, args=(), kwargs=None):
        # The probability product follows the softmax, the score product
        # precedes it.
        name = getattr(function, "__name__", None)
        if function is F.softmax:
            self.probabilities = True
        if name == "linear" and args[0].shape[-1] % 32 == 0:
            args = (mxint8(args[0], -1), mxint8(args[1], 1), *args[2:])
        elif name == "matmul":
            if args[0].shape[-1] % 32 == 0:
                key = mxint8(args[1], -2) if self.probabilities else split(args[1], -2)
                args = (mxint8(args[0], -1), key)
            self.probabilities = False
        elif name == "scaled_dot_product_attention":
            query, key, value, mask = args[:4]
            root = 1 / math.sqrt(math.sqrt(query.shape[-1]))
            scores = mxint8(query * root, -1) @ split(key.transpose(-2, -1) * root, -2)
            weights = torch.softmax(scores + mask, -1)
            return mxint8(weights, -1) @ mxint8(value, -2)
        elif name == "multi_head_attention_forward":
            dispatch = F.has_torch_function
            F.has_torch_function = lambda operands: False
            with self:
                result = function(*args, **(kwargs or {}))
            F.has_torch_function = dispatch
            return result
        return function(*args, **(kwargs or {}))


# ===----------------------------------------------------------------------===
# Evaluate MXINT8 in software on the questions of 200 typed-decision records:
# the weights of every Linear packed on K, its input and both operands of
# every attention product read through MXINT8. The quantization is rejected
# when the decision accuracy against the gold answers drops more than 1 point
# below FP32.
# ===----------------------------------------------------------------------===
model = laya.load()
tests, labels = decisions.load(200, 200)


def evaluate(products):
    """The decision accuracy and the logits of the markers."""
    logits = []
    with torch.no_grad():
        for batch in zip(*(value.split(16) for value in tests)):
            if products:
                with Products():
                    logits.append(model(*batch)[0])
            else:
                logits.append(model(*batch)[0])
    logits = torch.cat(logits)
    return (logits.argmax(1) == labels).float().mean().item() * 100, logits


fp32, reference = evaluate(False)
quantized, logits = evaluate(True)
# The logits of the markers that hold an option.
valid = tests[3]
error = ((logits - reference)[valid].norm() / reference[valid].norm()).item()
print(
    f"FP32 accuracy {fp32:.2f}%, MXINT8 accuracy {quantized:.2f}% on {len(labels)} "
    f"decisions, marker logits relative L2 error {error:.4f}"
)
if fp32 - quantized > 1:
    raise RuntimeError(
        f"MXINT8 accuracy {quantized:.2f}% is more than 1 below FP32 {fp32:.2f}%"
    )
