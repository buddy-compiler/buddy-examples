from importlib import import_module
from pathlib import Path

import torch
from torch.overrides import TorchFunctionMode

from compiler.frontend.quant import dequantize_mxint8, quantize_mxint8

model_dir = Path(__file__).resolve().parent
wikitext = import_module("models.models.qwen3-8b.data")
qwen3 = import_module("models.models.qwen3-8b.model")


def mxint8(values, axis):
    """Values through MXINT8 along `axis`, as the graph computes them."""
    return dequantize_mxint8(quantize_mxint8(values, axis), axis)


class Products(TorchFunctionMode):
    """Reads both operands of every matrix product through MXINT8 along its
    reduction axis; a product whose K is not a multiple of 32 stays FP32."""

    def __torch_function__(self, function, types, args=(), kwargs=None):
        # `a @ b` arrives as the matmul method of the tensor.
        if (
            getattr(function, "__name__", None) == "matmul"
            and args[0].shape[-1] % 32 == 0
        ):
            args = (mxint8(args[0], -1), mxint8(args[1], -2))
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
model = qwen3.load().model
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
