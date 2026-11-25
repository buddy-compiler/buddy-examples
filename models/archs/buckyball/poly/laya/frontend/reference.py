import math
import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode
from compiler.frontend.quant import quantize_mxint8, dequantize_mxint8
from frontend.integer_activations import IntegerActivations
from frontend.integer_softmax import softmax as integer_softmax


class Products(TorchFunctionMode):
    """Calls `linear(input, weight)` on the input of every Linear and
    `product(index, operands)` on every attention product, numbered in the
    order the model runs them, using the returned product directly. The
    attention of the head runs as the imported graph runs it: both operands
    of the scores take the square root of the softmax scale. A product of
    one column (the rotary frequencies) is a broadcast multiply in the graph,
    not a matrix product. The attention of the head reaches the mode as one
    multi_head_attention_forward call: its body runs under the mode, without
    dispatching to it again, so that its Linears and products reach it."""

    def __init__(self, linear, product, quantized_softmax=False):
        super().__init__()
        self.linear = linear
        self.product = product
        self.index = 0
        self.quantized_softmax = quantized_softmax

    def __torch_function__(self, function, types, args=(), kwargs=None):
        name = getattr(function, "__name__", None)
        if name == "linear":
            args = (self.linear(args[0], args[1]), *args[1:])
        elif name == "matmul" and args[0].shape[-1] > 1:
            result = self.product(self.index, args)
            self.index += 1
            return result
        elif name == "scaled_dot_product_attention":
            query, key, value, mask = args[:4]
            root = 1 / math.sqrt(math.sqrt(query.shape[-1]))
            scores = self.product(
                self.index, (query * root, key.transpose(-2, -1) * root)
            )
            weights = (
                integer_softmax(scores + mask)
                if self.quantized_softmax
                else torch.softmax(scores + mask, -1)
            )
            result = self.product(self.index + 1, (weights, value))
            self.index += 2
            return result
        elif name == "softmax" and self.quantized_softmax:
            dim = args[1] if len(args) > 1 else kwargs["dim"]
            if dim not in (-1, args[0].ndim - 1):
                raise ValueError("Expected last-axis attention softmax")
            return integer_softmax(args[0])
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


def divide(value, divisor):
    quotient = torch.div(value, divisor, rounding_mode="floor")
    twice = (value - quotient * divisor) * 2
    return quotient + ((twice > divisor) | ((twice == divisor) & ((quotient & 1) != 0)))


def integer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    if len(normalized_shape) != 1 or normalized_shape[0] % 512:
        raise ValueError("Integer normalization requires one complete 512-element axis")
    n = normalized_shape[0]
    values = torch.round(input * 65536).to(torch.int64)
    centered = values - divide(values.sum(-1, keepdim=True), n)
    variance = divide((centered * centered).sum(-1, keepdim=True), n) + round(
        eps * (1 << 32)
    )
    # Host reference only: integer comparisons correct the FP64 square-root
    # seed to the exact integer floor used by the instruction.
    root = torch.sqrt(variance.to(torch.float64)).to(torch.int64)
    root -= (root * root > variance).to(torch.int64)
    root += ((root + 1) * (root + 1) <= variance).to(torch.int64)
    result = divide(centered * 65536, root)
    if weight is not None:
        result = divide(result * torch.round(weight * 65536).to(torch.int64), 65536)
    if bias is not None:
        result += torch.round(bias * 65536).to(torch.int64)
    return mxint8(result.to(torch.float32) / 65536, -1)


class Norms(TorchFunctionMode):
    def __torch_function__(self, function, types, args=(), kwargs=None):
        if getattr(function, "__name__", None) == "layer_norm":
            return integer_norm(*args, **(kwargs or {}))
        return function(*args, **(kwargs or {}))


def evaluate(model, inputs):
    weights = set()

    def record(x, weight):
        weights.add(weight)
        return x

    with torch.no_grad(), Products(
        record, lambda index, operands: operands[0] @ operands[1]
    ):
        baseline = model(*inputs)
    for weight in weights:
        weight.data = mxint8(weight.detach(), 1)

    return baseline, quantized_forward(model, inputs)


def quantized_forward(model, inputs):
    """Evaluate one decision using the model's prepared MXINT8 weights."""

    def linear(x, weight):
        return mxint8(x, -1)

    def product(index, operands):
        x, w = operands
        return mxint8(x, -1) @ mxint8(w, -2)

    with torch.no_grad(), Products(
        linear, product, quantized_softmax=True
    ), Norms(), IntegerActivations():
        return model(*inputs)
