# ===- buddy-pytorch-calibrate.py -----------------------------------------
#
# SmolVLA quantization check: the policy as the quantized graph computes it
# (Linears and attention products in MXINT8, the patch convolution in INT8),
# rejected when the actions of the test case drift from FP32. Every
# activation scale is computed online, so nothing is calibrated:
# calibration.json records the accepted error.
#
# ===---------------------------------------------------------------------------

from importlib import import_module
from pathlib import Path
import json

import torch
from torch.overrides import TorchFunctionMode

from compiler.frontend.quant import dequantize_mxint8, quantize, quantize_mxint8

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
smolvla = import_module("models.models.smolvla-base.model")
model = smolvla.load()
# The vision encoder attends through explicit products, as the importer.
model.flow.vlm_with_expert.get_vlm_model().vision_model.config._attn_implementation = (
    "eager"
)

# The test case of the importer.
example = smolvla.inputs()
torch.manual_seed(0)
images = torch.rand(example[0].shape) * 2 - 1
lang_tokens = torch.randint(0, 1000, example[1].shape)
lang_masks = example[2]
state = torch.randn(example[3].shape)
noise = torch.randn(example[4].shape)
inputs = (images, lang_tokens, lang_masks, state, noise)


def mxint8(values, axis):
    """Values through MXINT8 along `axis`, as the graph computes them."""
    return dequantize_mxint8(quantize_mxint8(values, axis), axis)


def tokenwise(x, axis):
    """x as INT8 values with one scale per token: the largest magnitude
    along `axis` over 127."""
    step = x.abs().amax(axis, keepdim=True).clamp(min=1e-30) / 127
    return quantize(x, 1 / step) * step


class Products(TorchFunctionMode):
    """Reads both operands of every attention product through MXINT8 along
    its reduction axis; a product whose K is not a multiple of 32 stays
    FP32."""

    def __torch_function__(self, function, types, args=(), kwargs=None):
        # `a @ b` arrives as torch._C.TensorBase.matmul.
        if (
            getattr(function, "__name__", None) == "matmul"
            and args[0].shape[-1] % 32 == 0
        ):
            args = (mxint8(args[0], -1), mxint8(args[1], -2))
        return function(*args, **(kwargs or {}))


with torch.no_grad():
    fp32 = model(*inputs)

# ===----------------------------------------------------------------------===
# Run the quantized policy in software on the same inputs: the weights of the
# Linears packed in MXINT8 along K and their inputs quantized online; the
# Linears whose K is not a multiple of 32 and the patch convolution with
# INT8 weights (one scale per output channel) and INT8 inputs (one online
# scale per token, per pixel); biases FP32. Rejected when the
# relative L2 error of the actions exceeds 0.05.
# ===----------------------------------------------------------------------===
for module in model.modules():
    weight = module.weight.detach() if hasattr(module, "weight") else None
    if isinstance(module, torch.nn.Linear) and module.in_features % 32 == 0:
        # A new tensor: a weight may be shared with an embedding.
        module.weight = torch.nn.Parameter(mxint8(weight, 1))
        module.register_forward_pre_hook(lambda module, args: mxint8(args[0], -1))
    elif isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
        # The Linears whose K is not a multiple of 32 and the convolution:
        # INT8 with one scale per output channel and per token (pixel).
        axis = 1 if isinstance(module, torch.nn.Conv2d) else -1
        step = weight.abs().amax(list(range(1, weight.ndim)), keepdim=True) / 127
        module.weight = torch.nn.Parameter(quantize(weight, 1 / step) * step)
        module.register_forward_pre_hook(
            lambda module, args, axis=axis: tokenwise(args[0], axis)
        )
with torch.no_grad(), Products():
    quantized = model(*inputs)
error = ((quantized - fp32).norm() / fp32.norm()).item()
print(f"Quantized actions: relative L2 error {error:.4f} against FP32")
if error > 0.05:
    raise RuntimeError(
        f"Quantized actions drift from FP32: relative L2 error {error:.4f}"
    )
(output / "calibration.json").write_text(json.dumps({"error": error}, indent=2) + "\n")
