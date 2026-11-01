import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from compiler.frontend.quant import dequantize_mxint8, quantize_mxint8


def mxint8(x, axis):
    """x zero-padded along `axis` to a multiple of 32 and read through MXINT8
    along it, as the quantized graphs compute it."""
    axis %= x.ndim
    x = F.pad(x, (0, 0) * (x.ndim - 1 - axis) + (0, -x.shape[axis] % 32))
    return dequantize_mxint8(quantize_mxint8(x, axis), axis)


def two(x, axis):
    """x read as two MXINT8 terms along `axis`, as a doubled weight: its
    MXINT8 value plus the MXINT8 value of what that leaves out."""
    head = mxint8(x, axis)
    axis %= x.ndim
    x = F.pad(x, (0, 0) * (x.ndim - 1 - axis) + (0, head.shape[axis] - x.shape[axis]))
    return head + mxint8(x - head, axis)


def bf16(x, axis):
    """x rounded to BF16: the reference precision."""
    return x.bfloat16().float()


class Products(TorchFunctionMode):
    """Reads both operands of every matrix product through `read` along its
    reduction axis: the Linears, the attention products and the patch
    convolution, whose stride is its kernel, as the Linear of the patches.
    A weight among `doubled` is read as two MXINT8 terms. A weight is read
    once."""

    def __init__(self, read, doubled):
        super().__init__()
        self.read = read
        self.doubled = doubled
        self.weights = {}

    def weight(self, w):
        if w not in self.weights:
            read = two if w in self.doubled else self.read
            self.weights[w] = read(w.reshape(w.shape[0], -1), -1)
        return self.weights[w]

    def __torch_function__(self, function, types, args=(), kwargs=None):
        name = getattr(function, "__name__", None)
        # `a @ b` arrives as torch._C.TensorBase.matmul.
        if name == "matmul":
            return function(self.read(args[0], -1), self.read(args[1], -2))
        if name == "linear":
            x, w, b = args
            return F.linear(self.read(x, -1), self.weight(w), b)
        if name == "conv2d":
            x, w, b = args[:3]
            f, c, kh, kw = w.shape
            _, _, height, width = x.shape
            h, w_ = height // kh, width // kw
            rows = x.reshape(c, h, kh, w_, kw).permute(1, 3, 0, 2, 4)
            rows = rows.reshape(h * w_, c * kh * kw)
            out = F.linear(self.read(rows, -1), self.weight(w), b)
            return out.reshape(1, h, w_, f).permute(0, 3, 1, 2)
        return function(*args, **(kwargs or {}))
