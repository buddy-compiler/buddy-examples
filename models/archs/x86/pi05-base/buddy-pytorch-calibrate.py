# ===- buddy-pytorch-calibrate.py -----------------------------------------
#
# pi0.5 quantization check: the policy as the quantized graph computes it
# (MXINT8 or INT8 weights and two INT8 activation terms), rejected
# when its actions on SO100 frames drift from FP32. Every activation scale
# is computed online, so nothing is calibrated: calibration.json records the
# accepted error.
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
pi05 = import_module("models.models.pi05-base.model")
model = pi05.load()
# The vision encoder attends through explicit products, as the importer.
model.flow.paligemma_with_expert.paligemma.model.vision_tower.config._attn_implementation = (
    "eager"
)

# The evaluation inputs: the 100th frame of four SO100 pick-and-place
# episodes.
observation = import_module("models.models.pi05-base.observation")
evaluation = [observation.frame(episode) for episode in (40, 43, 46, 49)]


def mxint8(values, axis):
    """Values through MXINT8 along `axis`, as the graph computes them."""
    return dequantize_mxint8(quantize_mxint8(values, axis), axis)


def tokenwise(x, axis):
    """x as INT8 values with one scale per token: the largest magnitude
    along `axis` over 127."""
    step = x.abs().amax(axis, keepdim=True).clamp(min=1e-30) * (1 / 127)
    return quantize(x, 1 / step) * step


def split(x, axis):
    """x as the two INT8 terms the graph reads it with: its INT8 value plus
    the INT8 value of what that leaves out, one scale per token each."""
    head = tokenwise(x, axis)
    return head + tokenwise(x - head, axis)


class Products(TorchFunctionMode):
    """Read attention operands as two INT8 terms along the reduction axis.
    Products whose K is not a multiple of 32 stay FP32.
    """

    def __torch_function__(self, function, types, args=(), kwargs=None):
        # `a @ b` arrives as torch._C.TensorBase.matmul.
        if (
            getattr(function, "__name__", None) == "matmul"
            and args[0].shape[-1] % 32 == 0
        ):
            args = (split(args[0], -1), split(args[1], -2))
        return function(*args, **(kwargs or {}))


with torch.no_grad():
    fp32 = torch.stack([model(*inputs) for inputs in evaluation])

# ===----------------------------------------------------------------------===
# Check the same frames with MXINT8 Linear weights when K is divisible by 32,
# channelwise INT8 weights otherwise, and two online INT8 activation terms.
# Biases stay FP32. Reject actions whose relative L2 error exceeds 0.05.
# ===----------------------------------------------------------------------===
for module in model.modules():
    weight = module.weight.detach() if hasattr(module, "weight") else None
    if isinstance(module, torch.nn.Linear) and module.in_features % 32 == 0:
        # A new tensor: the token embeddings share their weight with lm_head.
        module.weight = torch.nn.Parameter(mxint8(weight, 1))
        module.register_forward_pre_hook(lambda module, args: split(args[0], -1))
    elif isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
        step = weight.abs().amax(list(range(1, weight.ndim)), keepdim=True) / 127
        module.weight = torch.nn.Parameter(quantize(weight, 1 / step) * step)
        if isinstance(module, torch.nn.Conv2d):
            module.register_forward_pre_hook(lambda module, args: split(args[0], 1))
        else:
            module.register_forward_pre_hook(lambda module, args: split(args[0], -1))
with torch.no_grad(), Products():
    quantized = torch.stack([model(*inputs) for inputs in evaluation])
error = ((quantized - fp32).norm() / fp32.norm()).item()
print(f"Quantized actions: relative L2 error {error:.4f} against FP32")
if error > 0.05:
    raise RuntimeError(
        f"Quantized actions drift from FP32: relative L2 error {error:.4f}"
    )
(output / "calibration.json").write_text(json.dumps({"error": error}, indent=2) + "\n")
