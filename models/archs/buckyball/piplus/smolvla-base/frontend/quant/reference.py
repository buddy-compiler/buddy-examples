import ctypes
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from numba import njit
from torch.overrides import TorchFunctionMode

from compiler.frontend.quant import dequantize_mxint8, quantize_mxint8

libm = ctypes.CDLL("libm.so.6")
newlib = ctypes.CDLL(
    str(Path(__file__).resolve().parents[7] / "result/lib/libnewlib-reference.so")
)
for name in ("exp", "sin", "cos", "tanh"):
    for suffix, dtype in (("", ctypes.c_double), ("f", ctypes.c_float)):
        function = getattr(libm, name + suffix)
        function.restype = dtype
        function.argtypes = [dtype]
newlib.tanhf.restype = ctypes.c_float
newlib.tanhf.argtypes = [ctypes.c_float]


@njit
def apply_scalar(function, values):
    result = np.empty_like(values)
    for i in range(values.size):
        result[i] = function(values[i])
    return result


def scalar(name, values):
    """Scalar C math, without changing its rounding through vector math."""
    suffix = "f" if values.dtype == torch.float32 else ""
    function = (
        newlib.tanhf if name == "tanh" and suffix else getattr(libm, name + suffix)
    )
    array = values.detach().contiguous().numpy()
    return torch.from_numpy(
        apply_scalar(function, array.reshape(-1)).reshape(array.shape)
    )


def mean(values, dim, keepdim=False):
    dims = [d % values.ndim for d in ([dim] if isinstance(dim, int) else dim)]
    total, count = values, 1
    for dim in dims:
        parts = total.movedim(dim, 0)
        summed = torch.zeros_like(parts[0])
        for part in parts:
            summed = summed + part
        total = summed.unsqueeze(dim)
        count *= values.shape[dim]
    total = total * torch.tensor(1 / count, dtype=values.dtype)
    return total if keepdim else total.squeeze(dims)


def mxint8(x, axis):
    """x zero-padded along `axis` to a multiple of 32 and read through MXINT8
    along it, as the quantized graphs compute it."""
    axis %= x.ndim
    x = F.pad(x, (0, 0) * (x.ndim - 1 - axis) + (0, -x.shape[axis] % 32))
    return dequantize_mxint8(quantize_mxint8(x, axis), axis)


def bf16(x, axis):
    """x rounded to BF16: the reference precision."""
    return x.bfloat16().float()


class Products(TorchFunctionMode):
    """Reads both operands of every matrix product through `read` along its
    reduction axis: the Linears, the attention products and the patch
    convolution, whose stride is its kernel, as the Linear of the patches.
    A weight among `doubled` is read as two MXINT8 terms, with their
    products added separately. A weight is read once."""

    def __init__(self, read, doubled):
        super().__init__()
        self.read = read
        self.doubled = doubled
        self.weights = {}

    def weight(self, w):
        if w not in self.weights:
            value = w.reshape(w.shape[0], -1)
            head = self.read(value, -1)
            tail = None
            if w in self.doubled:
                value = F.pad(value, (0, head.shape[-1] - value.shape[-1]))
                tail = mxint8(value - head, -1)
            self.weights[w] = head, tail
        return self.weights[w]

    def linear(self, x, w, bias):
        left = self.read(x, -1)
        head, tail = self.weight(w)
        if self.read is not mxint8:
            return F.linear(left, head, bias)
        result = self.product(left, head.transpose(-1, -2))
        if bias is not None:
            result = result + bias
        if tail is not None:
            result = result + self.product(left, tail.transpose(-1, -2))
        return result

    def product(self, left, right):
        # The Ball adds exact INT8 dot products in 32-element block order.
        if self.read is not mxint8:
            return left @ right
        result = left[..., :32] @ right[..., :32, :]
        for start in range(32, left.shape[-1], 32):
            result = (
                result
                + left[..., start : start + 32] @ right[..., start : start + 32, :]
            )
        return result

    def __torch_function__(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(function, "__name__", None)
        if name in ("sin", "cos", "exp", "tanh"):
            return scalar(name, args[0])
        if name == "__rpow__":
            return scalar(
                "exp", args[0] * torch.tensor(math.log(args[1]), dtype=args[0].dtype)
            )
        if name == "rsqrt":
            return torch.ones_like(args[0]) / torch.sqrt(args[0])
        if name == "silu":
            return args[0] / (1 + scalar("exp", -args[0]))
        if name == "mean":
            return mean(*args, **kwargs)
        if name == "softmax":
            dim = args[1] if len(args) > 1 else kwargs["dim"]
            values = args[0]
            exponentials = scalar("exp", values - values.amax(dim, keepdim=True))
            parts = exponentials.movedim(dim, 0)
            total = torch.zeros_like(parts[0])
            for part in parts:
                total = total + part
            return exponentials / total.unsqueeze(dim)
        if name == "layer_norm":
            values, shape = args[:2]
            weight = args[2] if len(args) > 2 else kwargs["weight"]
            bias = args[3] if len(args) > 3 else kwargs["bias"]
            epsilon = args[4] if len(args) > 4 else kwargs["eps"]
            dims = list(range(values.ndim - len(shape), values.ndim))
            centered = values - mean(values, dims, True)
            variance = mean(centered * centered, dims, True)
            result = centered * (
                torch.ones_like(variance) / torch.sqrt(variance + epsilon)
            )
            if weight is not None:
                result = result * weight
            if bias is not None:
                result = result + bias
            return result
        if name == "gelu":
            if kwargs["approximate"] != "tanh":
                raise ValueError("Expected tanh GELU")
            values = args[0]
            cubic = (values * values) * values
            argument = (values + cubic * 0.044715) * math.sqrt(2 / math.pi)
            return (values * 0.5) * (1 + scalar("tanh", argument))
        # `a @ b` arrives as torch._C.TensorBase.matmul.
        if name == "matmul":
            left, right = self.read(args[0], -1), self.read(args[1], -2)
            return self.product(left, right)
        if name == "linear":
            x, w, b = args
            return self.linear(x, w, b)
        if name == "conv2d":
            x, w, b = args[:3]
            f, c, kh, kw = w.shape
            _, _, height, width = x.shape
            h, w_ = height // kh, width // kw
            rows = x.reshape(c, h, kh, w_, kw).permute(1, 3, 0, 2, 4)
            rows = rows.reshape(h * w_, c * kh * kw)
            out = self.linear(rows, w, None)
            if b is not None:
                out = out + b
            return out.reshape(1, h, w_, f).permute(0, 3, 1, 2)
        return function(*args, **kwargs)
