import math
import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode
from compiler.frontend.quant import quantize_mxint8, dequantize_mxint8


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


def mxint8(value, axis):
    return dequantize_mxint8(quantize_mxint8(value, axis), axis)


def evaluate(model, inputs):
    weights = set()

    def record(x, weight):
        weights.add(weight)
        return x

    with torch.no_grad(), Products(record, lambda index, operands: operands):
        baseline = model(*inputs)
    for weight in weights:
        if weight.shape[1] % 32 == 0:
            weight.data = mxint8(weight.detach(), 1)

    def linear(x, weight):
        return mxint8(x, -1) if weight.shape[1] % 32 == 0 else x

    def product(index, operands):
        x, w = operands
        if x.shape[-1] % 32:
            return x, w
        x, head = mxint8(x, -1), mxint8(w, -2)
        return x, head if index % 2 else head + mxint8(w - head, -2)

    with torch.no_grad(), Products(linear, product):
        return baseline, model(*inputs)
