import math
import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode
from compiler.frontend.quant import quantize, quantize_bias


class Products(TorchFunctionMode):
    """Calls `linear(input, weight)` on the input of every Linear and
    `product(index, operands)` on every attention product, numbered in the
    order the model runs them, and computes with what they return. The
    attention of the head runs as the imported graph runs it: both operands
    of the scores take the square root of the softmax scale. A product of
    one column (the rotary frequencies) is a broadcast multiply in the graph,
    not a matrix product. The attention of the head reaches the mode as one
    multi_head_attention_forward call: its body runs under the mode, without
    dispatching to it again, so that its Linears and products reach it."""

    def __init__(self, linear, product):
        super().__init__()
        self.linear = linear
        self.product = product
        self.index = 0

    def __torch_function__(self, function, types, args=(), kwargs=None):
        name = getattr(function, "__name__", None)
        if name == "linear":
            args = (self.linear(args[0], args[1]), *args[1:])
        elif name == "matmul" and args[0].shape[-1] > 1:
            args = self.product(self.index, args)
            self.index += 1
        elif name == "scaled_dot_product_attention":
            query, key, value, mask = args[:4]
            root = 1 / math.sqrt(math.sqrt(query.shape[-1]))
            query, key = self.product(
                self.index, (query * root, key.transpose(-2, -1) * root)
            )
            weights = torch.softmax(query @ key + mask, -1)
            weights, value = self.product(self.index + 1, (weights, value))
            self.index += 2
            return weights @ value
        elif name == "multi_head_attention_forward":
            dispatch = F.has_torch_function
            F.has_torch_function = lambda operands: False
            with self:
                result = function(*args, **(kwargs or {}))
            F.has_torch_function = dispatch
            return result
        return function(*args, **(kwargs or {}))


def evaluate(model, calibration, inputs):
    """Evaluate the smoothed model with the recipe's calibrated INT8 operands."""
    names = {parameter: name for name, parameter in model.named_parameters()}
    parameters = dict(model.named_parameters())
    for name, parameter in parameters.items():
        module = name.rsplit(".", 1)[0]
        if not name.endswith("weight") or module not in calibration:
            continue
        step = parameter.detach().abs().amax(1, keepdim=True) / 127
        parameter.data = quantize(parameter.detach(), 1 / step) * step
        bias = parameters.get(name[: -len("weight")] + "bias")
        if bias is not None:
            bias_step = step.reshape(-1) * torch.tensor(calibration[module] / 127)
            bias.data = quantize_bias(bias.detach(), bias_step) * bias_step

    def linear(x, weight):
        step = torch.tensor(calibration[names[weight].rsplit(".", 1)[0]] / 127)
        return quantize(x, 1 / step) * step

    def product(index, operands):
        steps = [torch.tensor(calibration[f"matmul.{index}.{i}"] / 127) for i in (0, 1)]
        return tuple(quantize(x, 1 / step) * step for x, step in zip(operands, steps))

    with torch.no_grad(), Products(linear, product):
        return model(*inputs)
