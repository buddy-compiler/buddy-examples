"""The device program of the decoder in PyTorch: the quantized products of
frontend/quant/graph.py and the tensor-parallel split of
frontend/partition/graph.py, written over the model rather than read from
the partitioned graphs, so that it checks them.

Every Linear reads its input through MXINT8 along K. Fixed body weights
use the recipe's MXINT8 or native INT5 policy; the LM head stays MXINT8.
Native INT5 column outputs and row partials pass through MXINT8. The
Linears after the attention and in the FFN (o_proj, down_proj) read a value
the ranks hold in pieces: each rank multiplies its pieces of K, the reduce
hart adds the partial sums in rank order. An attention product reads both
operands through MXINT8 along K, the key of the score product as two MXINT8
terms.

The operations whose rounding depends on how they are computed follow the
compiled graphs on the hart CPUs: a mean sums its values in order and
multiplies by the FP32 reciprocal of their count, cos and sin are those of
the C library (the newlib routines of the harts are the glibc ones)."""

import ctypes

import numpy
import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from compiler.frontend.quant import dequantize_mxint8, quantize_mxint8
from frontend.integer_activations import IntegerActivations


def mxint8(values, axis):
    """Values through MXINT8 along `axis`, as the graph computes them."""
    return dequantize_mxint8(quantize_mxint8(values, axis), axis)


def two(values, axis):
    """Values as the two MXINT8 terms the graph reads the key with."""
    head = mxint8(values, axis)
    return head, mxint8(values - head, axis)


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


def round_shift(values, shifts):
    """Signed integer right shift with round to nearest, ties to even."""
    amount = shifts.clamp(0, 63)
    magnitude = values.abs()
    quotient = magnitude >> amount
    remainder = magnitude - (quotient << amount)
    halfway = torch.ones_like(amount, dtype=torch.int64) << (amount - 1).clamp_min(0)
    rounded = quotient + (
        (amount > 0)
        & ((remainder > halfway) | ((remainder == halfway) & ((quotient & 1) != 0)))
    )
    return torch.where(shifts >= 64, 0, rounded * values.sign())


def native_product(activation, weight, low, high):
    """Packed MXINT8 result of the INT5 Ball, including its shared G28 state."""
    a = activation.contiguous().view(torch.uint8)
    w = weight.contiguous().view(torch.uint8)
    blocks = w.shape[1] // 22
    k = blocks * 32
    n = w.shape[0]
    if w.ndim != 2 or w.shape[1] != blocks * 22 or n % 32:
        raise ValueError("Native INT5 requires complete 32-output groups")
    if a.shape[-1] != k + blocks or low % 32 or high % 32 or not 0 <= low < high <= k:
        raise ValueError(
            "Native INT5 requires an aligned K interval of the original packed weights"
        )
    rows = a.numel() // a.shape[-1]
    ac = a[..., :k].contiguous().view(torch.int8).reshape(rows, blocks, 32)
    ae = a[..., k:].reshape(rows, blocks).to(torch.int16)
    first, end = low // 32, high // 32
    interval = end - first
    words = w[:, : blocks * 20].reshape(n, blocks, 4, 5)[:, first:end].long()
    word = sum(words[..., i] << (8 * i) for i in range(5))
    codes = torch.stack([((word >> (5 * i)) & 31).to(torch.int8) for i in range(8)], -1)
    codes = ((codes ^ 16) - 16).reshape(n, interval, 32)
    factors = w[:, blocks * 20 : blocks * 21][:, first:end].long()
    exponents = w[:, blocks * 21 :][:, first:end].to(torch.int16)
    if (
        torch.any(codes == -16)
        or torch.any((factors < 1) | (factors > 137))
        or torch.any(exponents == 255)
        or torch.any(ae == 255)
    ):
        raise ValueError("Reserved native INT5 or MXINT8 encoding")
    # Each unscaled 32-code integer dot is below 2^24, hence exact in FP32.
    dots = torch.bmm(
        ac[:, first:end].float().permute(1, 0, 2), codes.float().permute(1, 2, 0)
    ).to(torch.int64)
    dots *= factors.T[:, None, :]
    sums = torch.zeros((rows, n // 32, 32), dtype=torch.int64)
    scale = torch.zeros((rows, n // 32), dtype=torch.int16)
    valid = torch.zeros_like(scale, dtype=torch.bool)
    for offset, block in enumerate(range(first, end)):
        products = dots[offset].reshape(rows, n // 32, 32)
        scales = (ae[:, block, None] + exponents[:, offset][None, :] - 258).reshape(
            rows, n // 32, 32
        )
        active = (products != 0).any(-1)
        incoming = torch.where(products != 0, scales, -32768).amax(-1)
        next_scale = torch.where(
            active, torch.where(valid, torch.maximum(scale, incoming), incoming), scale
        )
        sums = round_shift(sums, (next_scale - scale).clamp_min(0)[..., None])
        shift = 28 + scales - next_scale[..., None]
        terms = torch.where(
            shift >= 0, products << shift.clamp(0, 28), round_shift(products, -shift)
        )
        sums += terms
        valid = (sums != 0).any(-1)
        scale = torch.where(valid, next_scale, 0)
    maximum = sums.abs().amax(-1)
    residual = maximum.clone()
    highest = torch.zeros_like(maximum)
    for bits in (32, 16, 8, 4, 2, 1):
        above = residual >= (1 << bits)
        residual = torch.where(above, residual >> bits, residual)
        highest += above * bits
    exponent = torch.where(maximum == 0, 0, (highest + scale - 28 - 6).clamp(-127, 127))
    shift = exponent - scale + 28
    left = (-shift).clamp(0, 6)[..., None]
    saturated = (shift[..., None] < -6) | (sums.abs() > (127 >> left))
    result = torch.where(
        shift[..., None] >= 0,
        round_shift(sums, shift[..., None]),
        torch.where(saturated, sums.sign() * 127, sums << left),
    )
    codes = result.clamp(-127, 127).to(torch.int8).reshape(*a.shape[:-1], n)
    encoded = (
        (exponent + 127)
        .to(torch.uint8)
        .view(torch.int8)
        .reshape(*a.shape[:-1], n // 32)
    )
    return torch.cat([codes, encoded], -1)


class Program(TorchFunctionMode):
    """The products of the decoder as the device program computes them, over
    `ranks` ranks; `rows` are the weights of the Linears split along K."""

    probabilities = False

    def __init__(self, rows, ranks, native_weights):
        super().__init__()
        self.rows = rows
        self.ranks = ranks
        self.native_weights = native_weights

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
            if id(weight) in self.native_weights:
                activation = quantize_mxint8(x, -1)
                packed = self.native_weights[id(weight)]
                if not row:
                    return dequantize_mxint8(
                        native_product(activation, packed, 0, x.shape[-1]), -1
                    )
                total = None
                for low, high in pieces(x.shape[-1], self.ranks):
                    partial = dequantize_mxint8(
                        native_product(activation, packed, low, high), -1
                    )
                    total = partial if total is None else total + partial
                return total
            x = mxint8(x, -1)
            if not row:
                result = linear(x, weight)
                return result
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


def run(decoder, tokens, start, keys, values, ranks, native_weights):
    """Run with the recipe's prequantized Linear weights; return logits and KV."""
    rows = []
    for layer in decoder.model.model.layers:
        rows += [layer.self_attn.o_proj.weight, layer.mlp.down_proj.weight]
    activations = IntegerActivations()
    with torch.no_grad(), Program(rows, ranks, native_weights), activations:
        return decoder(tokens, start, keys, values)
