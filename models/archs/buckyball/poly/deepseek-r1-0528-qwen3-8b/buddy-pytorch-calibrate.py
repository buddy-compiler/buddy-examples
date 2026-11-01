from importlib import import_module
from pathlib import Path

import torch
from torch.overrides import TorchFunctionMode

from compiler.frontend.quant import dequantize_mxint8, quantize_mxint8

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
wikitext = import_module("models.models.deepseek-r1-0528-qwen3-8b.data")
deepseek = import_module("models.models.deepseek-r1-0528-qwen3-8b.model")


def mxint8(values, axis):
    """Values through MXINT8 along `axis`, as the graph computes them."""
    return dequantize_mxint8(quantize_mxint8(values, axis), axis)


def split(values, axis):
    """Values as the two MXINT8 terms the graph reads the key and the
    probabilities with."""
    head = mxint8(values, axis)
    return head + mxint8(values - head, axis)


class Products(TorchFunctionMode):
    """Reads both operands of every matrix product through MXINT8 along its
    reduction axis, the key of the score product q k^T and the probabilities
    of the product P v as two MXINT8 terms; a product whose K is not a
    multiple of 32 stays FP32."""

    probabilities = False

    def __torch_function__(self, function, types, args=(), kwargs=None):
        # `a @ b` arrives as the matmul method of the tensor; the probability
        # product follows the softmax, the score product precedes it.
        if function is torch.nn.functional.softmax:
            self.probabilities = True
        if getattr(function, "__name__", None) == "matmul":
            if args[0].shape[-1] % 32 == 0:
                if self.probabilities:
                    args = (split(args[0], -1), mxint8(args[1], -2))
                else:
                    args = (mxint8(args[0], -1), split(args[1], -2))
            self.probabilities = False
        return function(*args, **(kwargs or {}))


# ===----------------------------------------------------------------------===
# The test text of WikiText-2 in windows of the 256 tokens of the KV cache.
# ===----------------------------------------------------------------------===
windows = wikitext.windows(32)


def perplexity(model, products):
    losses = []
    with torch.no_grad():
        for window in windows.split(8):
            if products:
                with Products():
                    loss = model(input_ids=window, labels=window).loss
            else:
                loss = model(input_ids=window, labels=window).loss
            losses.append(loss)
    return torch.stack(losses).mean().exp().item()


# ===----------------------------------------------------------------------===
# Evaluate MXINT8 in software: the weights of every Linear packed offline, its
# input and both operands of every attention product quantized online. The
# quantization is rejected when the perplexity rises more than 1% over FP32.
# ===----------------------------------------------------------------------===
model = deepseek.load().model
fp32 = perplexity(model, False)
for module in model.modules():
    if isinstance(module, torch.nn.Linear):
        module.weight = torch.nn.Parameter(mxint8(module.weight.detach(), 1))
        module.register_forward_pre_hook(
            lambda module, inputs: (mxint8(inputs[0], -1),)
        )
quantized = perplexity(model, True)
print(f"FP32 perplexity {fp32:.3f}, MXINT8 perplexity {quantized:.3f}")
if quantized > fp32 * 1.01:
    raise RuntimeError(
        f"MXINT8 perplexity {quantized:.3f} is more than 1% above FP32 {fp32:.3f}"
    )
