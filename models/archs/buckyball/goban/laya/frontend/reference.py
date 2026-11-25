from models.archs.buckyball.goban.exp import gelu, softmax
import ctypes
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode
from compiler.frontend.quant import quantize, quantize_bias


class Products(TorchFunctionMode):
    """Computes every Linear as `linear(input, weight, bias)` and every
    attention product as `product(index, operands)`, numbered in the order
    the model runs them; every LayerNorm with its sums in order, and sin and
    cos with the newlib library at `newlib`, as the program computes them. The
    attention of the head runs as the imported graph runs it: both operands
    of the scores take the square root of the softmax scale. A product of
    one column (the rotary frequencies) is a broadcast multiply in the graph,
    not a matrix product. The attention of the head reaches the mode as one
    multi_head_attention_forward call: its body runs under the mode, without
    dispatching to it again, so that its Linears and products reach it."""

    def __init__(self, linear, product, newlib):
        super().__init__()
        self.linear = linear
        self.product = product
        self.index = 0
        library = ctypes.CDLL(newlib)
        self.newlib = {name: getattr(library, f"{name}f") for name in ("sin", "cos")}
        for function in self.newlib.values():
            function.restype = ctypes.c_float
            function.argtypes = [ctypes.c_float]

    def __torch_function__(self, function, types, args=(), kwargs=None):
        name = getattr(function, "__name__", None)
        if name == "softmax":
            return softmax(args[0], args[1] if len(args) > 1 else kwargs["dim"])
        if name == "gelu":
            return gelu(args[0])
        if name in self.newlib:
            values = args[0].contiguous().reshape(-1).tolist()
            function = self.newlib[name]
            return torch.tensor([function(v) for v in values]).reshape(args[0].shape)
        if name == "layer_norm":
            return layer_norm(*args, **(kwargs or {}))
        if name == "linear":
            bias = args[2] if len(args) > 2 else (kwargs or {}).get("bias")
            return self.linear(args[0], args[1], bias)
        if name == "matmul" and args[0].shape[-1] > 1:
            self.index += 1
            return self.product(self.index - 1, args)
        elif name == "scaled_dot_product_attention":
            query, key, value, mask = args[:4]
            root = 1 / math.sqrt(math.sqrt(query.shape[-1]))
            scores = self.product(
                self.index, (query * root, key.transpose(-2, -1) * root)
            )
            weights = softmax(scores + mask, -1)
            result = self.product(self.index + 1, (weights, value))
            self.index += 2
            return result
        elif name == "multi_head_attention_forward":
            dispatch = F.has_torch_function
            F.has_torch_function = lambda operands: False
            with self:
                result = function(*args, **(kwargs or {}))
            F.has_torch_function = dispatch
            return result
        return function(*args, **(kwargs or {}))


def layer_norm(input, shape, weight=None, bias=None, eps=1e-5):
    """LayerNorm over the last dimension, each sum in order in FP32, times the
    inverse of the count; centered, times 1 / sqrt(variance + eps)."""
    x = input.detach().contiguous().numpy()
    inverse = np.float32(1) / np.float32(x.shape[-1])
    mean = np.add.accumulate(x, axis=-1, dtype=np.float32)[..., -1:] * inverse
    centered = x - mean
    variance = (
        np.add.accumulate(centered * centered, axis=-1, dtype=np.float32)[..., -1:]
        * inverse
    )
    result = centered * (np.float32(1) / np.sqrt(variance + np.float32(eps)))
    if weight is not None:
        result = result * weight.detach().numpy()
    if bias is not None:
        result = result + bias.detach().numpy()
    return torch.from_numpy(result)


def evaluate(model, calibration, inputs, newlib):
    """Evaluate the smoothed model with the recipe's calibrated INT8 operands."""
    names = {parameter: name for name, parameter in model.named_parameters()}
    parameters = dict(model.named_parameters())
    weights = {}
    for name, parameter in parameters.items():
        module = name.rsplit(".", 1)[0]
        if not name.endswith("weight") or module not in calibration:
            continue
        step = parameter.detach().abs().amax(1, keepdim=True) / 127
        weights[module] = (quantize(parameter.detach(), 1 / step), step.reshape(-1))
        parameter.data = weights[module][0] * step
        bias = parameters.get(name[: -len("weight")] + "bias")
        if bias is not None:
            bias_step = step.reshape(-1) * torch.tensor(calibration[module] / 127)
            bias.data = quantize_bias(bias.detach(), bias_step) * bias_step

    # As the program computes them: INT8 codes, an exact INT32 sum, converted
    # to FP32 and scaled by the product of both steps; a bias decoded apart
    # and added after.
    def linear(x, weight, bias):
        module = names[weight].rsplit(".", 1)[0]
        step = torch.tensor(calibration[module] / 127)
        weight_codes, weight_step = weights[module]
        codes = quantize(x, 1 / step).double() @ weight_codes.double().T
        result = codes.float() * (weight_step * step)
        return result if bias is None else result + bias

    def product(index, operands):
        steps = [torch.tensor(calibration[f"matmul.{index}.{i}"] / 127) for i in (0, 1)]
        codes = [quantize(x, 1 / step).double() for x, step in zip(operands, steps)]
        return (codes[0] @ codes[1]).float() * (steps[0] * steps[1])

    with torch.no_grad(), Products(linear, product, newlib):
        return model(*inputs)
