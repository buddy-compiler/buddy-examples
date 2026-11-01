"""The device program of the decoder in PyTorch: the quantized products of
frontend/quant/graph.py and the tensor-parallel split of
frontend/partition/graph.py, written over the model rather than read from
the partitioned graphs, so that it checks them.

Every Linear reads its weight and its input through MXINT8 along K. The
Linears after the attention and in the FFN (o_proj, down_proj) read a value
the ranks hold in pieces: each rank multiplies its pieces of K, the reduce
core adds the partial sums in rank order. An attention product reads both
operands through MXINT8 along K, the key of the score product as two MXINT8
terms.

The operations whose rounding depends on how they are computed follow the
compiled graphs on the core CPUs: a mean sums its values in order and
multiplies by the FP32 reciprocal of their count, cos and sin are those of
the C library (the newlib routines of the cores are the glibc ones)."""

import ctypes

import numpy
import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from compiler.frontend.quant import dequantize_mxint8, quantize_mxint8


def mxint8(values, axis):
    """Values through MXINT8 along `axis`, as the graph computes them."""
    return dequantize_mxint8(quantize_mxint8(values, axis), axis)


def two(values, axis):
    """Values as the two MXINT8 terms the graph reads the key with."""
    head = mxint8(values, axis)
    return head + mxint8(values - head, axis)


libm = ctypes.CDLL("libm.so.6")
for _name in ("cosf", "sinf", "expf", "tanhf"):
    getattr(libm, _name).restype = ctypes.c_float
    getattr(libm, _name).argtypes = [ctypes.c_float]


def libc(name, values):
    """`name` of the C library on every FP32 value."""
    function = getattr(libm, name)
    flat = values.detach().reshape(-1).tolist()
    return torch.tensor(
        numpy.array([function(v) for v in flat], dtype=numpy.float32)
    ).reshape(values.shape)


def mean(values, dims, keepdim=False):
    """The mean of the compiled graph: the values summed in order along each
    axis, times the FP32 reciprocal of their count."""
    dims = [d % values.ndim for d in ([dims] if isinstance(dims, int) else dims)]
    total = values
    count = 1
    for d in dims:
        moved = total.movedim(d, 0)
        accumulated = torch.zeros_like(moved[0])
        for index in range(moved.shape[0]):
            accumulated = accumulated + moved[index]
        total = accumulated.unsqueeze(d)
        count *= values.shape[d]
    result = total * torch.tensor(1.0 / count, dtype=torch.float32)
    return result if keepdim else result.squeeze(dims)


def pieces(k, ranks):
    """The K ranges of the ranks: MXINT8 blocks dealt out, the first ranks
    one more when they do not divide evenly."""
    base, extra = divmod(k // 32, ranks)
    result = []
    for rank in range(ranks):
        low = 32 * (rank * base + min(rank, extra))
        result.append((low, low + 32 * (base + (rank < extra))))
    return result


def linear(x, weight):
    """MXINT8's exact 32-code dot products, accumulated in block order.

    Within a block both operands have one power-of-two scale, so this FP32
    dot product is exact (the integer sum fits 24 bits). Across blocks the
    additions round to FP32, as the Ball does; a whole-K GEMM may reorder them.
    """
    result = x.new_zeros((*x.shape[:-1], weight.shape[0]))
    for low in range(0, x.shape[-1], 32):
        result = result + F.linear(x[..., low : low + 32], weight[:, low : low + 32])
    return result


class Program(TorchFunctionMode):
    """The products of the decoder as the device program computes them, over
    `ranks` ranks; `rows` are the weights of the Linears split along K."""

    probabilities = False

    def __init__(self, rows, ranks):
        super().__init__()
        self.rows = rows
        self.ranks = ranks

    def __torch_function__(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if function is F.softmax:
            self.probabilities = True
        name = getattr(function, "__name__", None)
        if name in ("cos", "sin", "tanh"):
            return libc(f"{name}f", args[0])
        if name == "rsqrt":
            return torch.ones_like(args[0]) / torch.sqrt(args[0])
        if name == "silu":
            return args[0] / (1 + libc("expf", -args[0]))
        if name == "softmax":
            self.probabilities = True
            dim = args[1] if len(args) > 1 else kwargs["dim"]
            values = args[0]
            exponentials = libc("expf", values - values.amax(dim, keepdim=True))
            parts = exponentials.movedim(dim, 0)
            total = torch.zeros_like(parts[0])
            for part in parts:
                total = total + part
            return exponentials / total.unsqueeze(dim)
        if name == "mean":
            return mean(*args, **kwargs)
        if name == "linear":
            x, weight = args[0], args[1]
            bias = args[2] if len(args) > 2 else kwargs.get("bias")
            if bias is not None:
                raise ValueError("a Linear with a bias")
            row = any(weight is w for w in self.rows)
            x, weight = mxint8(x, -1), mxint8(weight, 1)
            if not row:
                return linear(x, weight)
            total = None
            for low, high in pieces(x.shape[-1], self.ranks):
                partial = linear(x[..., low:high], weight[:, low:high])
                total = partial if total is None else total + partial
            return total
        if name == "matmul" and args[0].shape[-1] % 32 == 0:
            key = mxint8(args[1], -2) if self.probabilities else two(args[1], -2)
            query = mxint8(args[0], -1)
            # MXINT8 accumulates 32-code blocks; FP32 accumulates in K order.
            block = 32 if self.probabilities else 1
            self.probabilities = False
            result = function(query[..., :block], key[..., :block, :])
            for start in range(block, query.shape[-1], block):
                result = result + function(
                    query[..., start : start + block],
                    key[..., start : start + block, :],
                )
            return result
        return function(*args, **kwargs)


def run(decoder, tokens, start, keys, values, ranks):
    """The outputs of the device program: the logits and the updated cache."""
    rows = []
    for layer in decoder.model.model.layers:
        rows += [layer.self_attn.o_proj.weight, layer.mlp.down_proj.weight]
    with torch.no_grad(), Program(rows, ranks):
        return decoder(tokens, start, keys, values)
