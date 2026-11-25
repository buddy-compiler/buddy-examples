"""The device program of the decoder in PyTorch: the quantized products of
frontend/quant/graph.py and the tensor-parallel split of
frontend/partition/graph.py, written over the model rather than read from
the partitioned graphs, so that it checks them.

Every Linear reads its weight and its input through MXINT8 along K. The
Linears after the attention and in the FFN (o_proj, down_proj) read a value
the ranks hold in pieces: each rank multiplies its pieces of K, the reduce
hart adds the partial sums in rank order. An attention product reads both
operands through MXINT8 along K, the key of the score product as two MXINT8
terms.

The operations whose rounding depends on how they are computed follow the
compiled graphs: a mean sums its values in order and multiplies by the FP32
reciprocal of their count, cos, sin and powf are those of the C library on
the hart CPUs (the newlib routines of the harts are the glibc ones). The
exponential of the softmax and the hyperbolic tangent of GELU and of the
logit cap run in the operators of the threads, where the poly hart's
OutlineOperators expands them into MLIR's polynomial approximations with
fused multiply-adds: `exp` and `tanh` here are those operators, operation
by operation."""

import ctypes
import math

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
    return head, mxint8(values - head, axis)


libm = ctypes.CDLL("libm.so.6")
for function in (libm.cosf, libm.sinf):
    function.restype = ctypes.c_float
    function.argtypes = [ctypes.c_float]
libm.powf.restype = ctypes.c_float
libm.powf.argtypes = [ctypes.c_float, ctypes.c_float]


def libc(function, values, *arguments):
    """`function` of the C library on every FP32 value."""
    flat = values.detach().reshape(-1).tolist()
    return torch.tensor(
        numpy.array([function(v, *arguments) for v in flat], dtype=numpy.float32)
    ).reshape(values.shape)


def fma(a, b, c):
    """a * b + c rounded once to FP32, as math.fma: the product of two FP32
    values is exact in FP64, and their sum with c, rounded to odd in FP64,
    rounds to FP32 as the exact sum does."""
    a, b, c = (torch.as_tensor(v, dtype=torch.float32) for v in (a, b, c))
    product = a.double() * b.double()
    addend = c.double().expand_as(product)
    total = product + addend
    part = total - product
    error = (product - (total - part)) + (addend - part)
    even = (total.contiguous().view(torch.int64) & 1) == 0
    toward = torch.where(error > 0, torch.inf, -torch.inf).to(torch.float64)
    return torch.where(
        (error != 0) & even, torch.nextafter(total, toward), total
    ).float()


def f32(value):
    return torch.tensor(value, dtype=torch.float32)


def tanh(x):
    """math.tanh of the operators: x clamped to +-7.99881172; itself below
    0.0004, else an odd polynomial of degree 13 over an even one of degree 6,
    both in fused multiply-adds."""
    clamped = torch.where(x < f32(7.99881172), x, f32(7.99881172))
    clamped = torch.where(clamped > f32(-7.99881172), clamped, f32(-7.99881172))
    square = clamped * clamped
    p = fma(square, f32(-2.76076837e-16), f32(2.00018794e-13))
    for alpha in (
        -8.60467184e-11,
        5.12229725e-8,
        1.48572235e-5,
        6.37261954e-4,
        0.00489352457,
    ):
        p = fma(square, p, f32(alpha))
    q = fma(square, f32(1.19825836e-6), f32(1.18534706e-4))
    for beta in (0.00226843474, 0.00489352504):
        q = fma(square, q, f32(beta))
    return torch.where(x.abs() < f32(4.0e-04), clamped, (clamped * p) / q)


def exp(x):
    """math.exp of the operators: x clamped to [-87.8, 88.8], k the floor of
    x log2(e) + 0.5 within +-127, a polynomial of x - k ln 2 (ln 2 in two
    terms) in fused multiply-adds, times 2^k (zero at -127)."""
    x = torch.where(x >= f32(-87.8), x, f32(-87.8))
    x = torch.where(x <= f32(88.8), x, f32(88.8))
    k = torch.floor(fma(x, f32(1.44269502), f32(0.5)))
    k = torch.where(k >= f32(-127.0), k, f32(-127.0))
    k = torch.where(k <= f32(127.0), k, f32(127.0))
    r = fma(f32(-0.693359375), k, x)
    r = fma(f32(2.12194442e-4), k, r)
    p = fma(r, f32(1.98756912e-4), f32(0.00139819994))
    for coefficient in (0.00833345205, 0.0416657962, 0.166666657, 0.5):
        p = fma(p, r, f32(coefficient))
    y = fma(p, r * r, r) + f32(1.0)
    return y * ((k.to(torch.int32) + 127) << 23).view(torch.float32)


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
        if name in ("cos", "sin"):
            return libc(getattr(libm, f"{name}f"), args[0])
        if name == "tanh":
            return tanh(args[0])
        if name == "gelu":
            if kwargs["approximate"] != "tanh":
                raise ValueError("Gemma requires tanh GELU")
            # The exported scalar expression rounds each FP32 operation;
            # PyTorch's fused GELU can change subsequent MXINT8 quantization.
            x = args[0]
            inner = (x + ((x * x) * x) * 0.044715) * math.sqrt(2 / math.pi)
            return (x * 0.5) * (1 + tanh(inner))
        if name == "pow" and isinstance(args[1], float) and args[1] == -0.5:
            return libc(libm.powf, args[0], -0.5)
        if name == "rsqrt":
            return torch.ones_like(args[0]) / torch.sqrt(args[0])
        if name == "softmax":
            self.probabilities = True
            dim = args[1] if len(args) > 1 else kwargs["dim"]
            values = args[0]
            exponentials = exp(values - values.amax(dim, keepdim=True))
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
            x = mxint8(x, -1)
            if not row:
                return linear(x, weight)
            total = None
            for low, high in pieces(x.shape[-1], self.ranks):
                partial = linear(x[..., low:high], weight[:, low:high])
                total = partial if total is None else total + partial
            return total
        if name == "matmul" and args[0].shape[-1] % 32 == 0:
            keys = (mxint8(args[1], -2),) if self.probabilities else two(args[1], -2)
            query = mxint8(args[0], -1)
            self.probabilities = False
            partials = []
            for key in keys:
                result = function(query[..., :32], key[..., :32, :])
                for start in range(32, query.shape[-1], 32):
                    result = result + function(
                        query[..., start : start + 32],
                        key[..., start : start + 32, :],
                    )
                partials.append(result)
            return partials[0] if len(partials) == 1 else partials[0] + partials[1]
        return function(*args, **kwargs)


def run(decoder, tokens, start, keys, values, ranks):
    """Run with the recipe's prequantized Linear weights; return logits and
    KV."""
    rows = []
    for layer in decoder.model.model.layers:
        rows += [layer.self_attn.o_proj.weight, layer.mlp.down_proj.weight]
    with torch.no_grad(), Program(rows, ranks):
        return decoder(tokens, start, keys, values)
