import ctypes
import math

import numpy as np
import torch
import torch.nn.functional as F
from numba import njit
from torch.overrides import TorchFunctionMode

from compiler.frontend.quant import dequantize_mxint8, quantize_mxint8

libm = ctypes.CDLL("libm.so.6")
for name in ("exp", "sin", "cos", "tanh"):
    for suffix, dtype in (("", ctypes.c_double), ("f", ctypes.c_float)):
        function = getattr(libm, name + suffix)
        function.restype = dtype
        function.argtypes = [dtype]


@njit
def apply_scalar(function, values):
    result = np.empty_like(values)
    for i in range(values.size):
        result[i] = function(values[i])
    return result


def lanes(parts):
    """The sum of `parts` as LayerNormBall and SoftmaxBall add the values of
    a row: four lane sums in increasing element order, then
    (s0 + s1) + (s2 + s3)."""
    sums = [torch.zeros_like(parts[0]) for _ in range(4)]
    for index, part in enumerate(parts):
        sums[index % 4] = sums[index % 4] + part
    return (sums[0] + sums[1]) + (sums[2] + sums[3])


# SoftmaxBall's 2^f on [0, 1): 1 + f (c1 + f (c2 + ... f c6)), in FP32.
exp2_coefficients = (
    0.6931471805599453,
    0.2402265069591007,
    0.0555041086648216,
    0.0096181291076285,
    0.0013333558146428,
    0.0001540353039338,
)
log2e = 1.4426950408889634


def ball_softmax(values, dim):
    """The softmax of SoftmaxBall, operation for operation in FP32: 2^u of
    u = (x - max) log2(e) as 2^floor(u) times the polynomial of its
    fraction (zero below -126), the lane sum, one reciprocal."""
    u = (values - values.amax(dim, keepdim=True)) * torch.tensor(
        log2e, dtype=values.dtype
    )
    n = torch.floor(u)
    f = u - n
    p = torch.full_like(f, exp2_coefficients[5])
    for coefficient in reversed(exp2_coefficients[:5]):
        p = p * f + torch.tensor(coefficient, dtype=values.dtype)
    p = p * f + 1.0
    exponentials = torch.where(
        u < -126.0,
        torch.zeros_like(p),
        torch.ldexp(p, n.clamp(min=-126.0).to(torch.int32)),
    )
    total = lanes(exponentials.movedim(dim, 0))
    return exponentials * (torch.ones_like(total) / total).unsqueeze(dim)


def f32(value):
    return torch.tensor(value, dtype=torch.float32)


def fused(a, b, c):
    """a b + c rounded once: the fused multiply-add of the GP domain."""
    return (a.double() * b.double() + c.double()).float()


def operator_tanh(values):
    """tanh as an operator of the GP domain computes it: the rational
    polynomial the compiler writes for math.tanh (MLIR's TanhApproximation),
    operation for operation in FP32."""
    bound = f32(7.99881172180175781)
    x = torch.maximum(torch.minimum(values, bound), -bound)
    x2 = x * x
    p = fused(x2, f32(-2.76076847742355e-16), f32(2.00018790482477e-13))
    for alpha in (
        -8.60467152213735e-11,
        5.12229709037114e-08,
        1.48572235717979e-05,
        6.37261928875436e-04,
        4.89352455891786e-03,
    ):
        p = fused(x2, p, f32(alpha))
    p = x * p
    q = fused(x2, f32(1.19825839466702e-06), f32(1.18534705686654e-04))
    q = fused(x2, q, f32(2.26843463243900e-03))
    q = fused(x2, q, f32(4.89352518554385e-03))
    return torch.where(values.abs() < f32(0.0004), x, p / q)


def operator_exp(values):
    """exp as an operator of the GP domain computes it: the polynomial the
    compiler writes for math.exp (MLIR's ExpApproximation, from Cephes),
    operation for operation in FP32."""
    x = torch.maximum(torch.minimum(values, f32(88.8)), f32(-87.8))
    n = torch.floor(fused(x, f32(1.44269504088896341), f32(0.5)))
    n = torch.maximum(torch.minimum(n, f32(127.0)), f32(-127.0))
    x = fused(f32(-0.693359375), n, x)
    x = fused(f32(2.12194440e-4), n, x)
    z = fused(x, f32(1.9875691500e-4), f32(1.3981999507e-3))
    for coefficient in (
        8.3334519073e-3,
        4.1665795894e-2,
        1.6666665459e-1,
        5.0000001201e-1,
    ):
        z = fused(z, x, f32(coefficient))
    z = fused(z, x * x, x)
    z = 1.0 + z
    return torch.where(
        n == -127.0, torch.zeros_like(z), torch.ldexp(z, n.to(torch.int32))
    )


def mean(values, dim, keepdim=False):
    dims = [d % values.ndim for d in ([dim] if isinstance(dim, int) else dim)]
    total, count = values, 1
    for dim in dims:
        total = lanes(total.movedim(dim, 0)).unsqueeze(dim)
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
    convolution, whose stride is its operator, as the Linear of the patches.
    A weight among `doubled` is read as two MXINT8 terms, with their
    products added separately. A weight is read once. The FP32 tanh is the
    tanhf of the newlib library at `newlib`, as the harts compute it; the
    tanh of a GELU and the exp of a SiLU are those of the operators of the
    GP domain (operator_tanh, operator_exp); a softmax and the sums of a
    normalization are SoftmaxBall's and LayerNormBall's, operation for
    operation."""

    def __init__(self, read, doubled, newlib, whole):
        super().__init__()
        self.read = read
        self.doubled = doubled
        self.whole = whole
        self.weights = {}
        # The results of the Linears, to sum a residual with (the last few).
        self.linears = []
        self.tanhf = ctypes.CDLL(newlib).tanhf
        self.tanhf.restype = ctypes.c_float
        self.tanhf.argtypes = [ctypes.c_float]

    def scalar(self, name, values):
        """Scalar C math, without changing its rounding through vector math."""
        suffix = "f" if values.dtype == torch.float32 else ""
        function = (
            self.tanhf if name == "tanh" and suffix else getattr(libm, name + suffix)
        )
        array = values.detach().contiguous().numpy()
        return torch.from_numpy(
            apply_scalar(function, array.reshape(-1)).reshape(array.shape)
        )

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

    def linear(self, x, w, bias, start=None):
        """A Linear as the threads sum it (the products of
        convert-linalg-to-acc): K goes in pieces of the largest multiple of
        32 up to 1024 that divides it, and each piece adds the block terms of
        the weight's first MXINT8 term, then those of its second. In a graph
        that runs whole on one hart (`whole`) each element starts at `start`
        (the residual the result is added to) plus the bias (FuseBias.cpp);
        in a graph cut over the harts the sums start from zero and the bias
        is added after."""
        left = self.read(x, -1)
        head, tail = self.weight(w)
        if self.read is not mxint8:
            return F.linear(left, head, bias)
        result = start
        if self.whole and bias is not None:
            result = bias if start is None else start + bias
        width = left.shape[-1]
        piece = 1024
        while width % piece:
            piece -= 32
        for begin in range(0, width, piece):
            part = left[..., begin : begin + piece]
            result = self.product(
                part, head[..., begin : begin + piece].transpose(-1, -2), result
            )
            if tail is not None:
                result = self.product(
                    part, tail[..., begin : begin + piece].transpose(-1, -2), result
                )
        if bias is not None and not self.whole:
            result = result + bias
        if self.whole and start is None:
            self.linears.append((result, x, w, bias))
        return result

    def product(self, left, right, result=None):
        # The Ball adds exact INT8 dot products in 32-element block order.
        if self.read is not mxint8:
            return left @ right
        for start in range(0, left.shape[-1], 32):
            term = left[..., start : start + 32] @ right[..., start : start + 32, :]
            result = term if result is None else result + term
        return result

    def residual(self, left, right):
        """`left + right` when one of them is a Linear's result: the Linear
        summed from the other."""
        for value, other in ((left, right), (right, left)):
            if not torch.is_tensor(value) or not torch.is_tensor(other):
                continue
            for result, x, w, bias in self.linears:
                if (
                    result.data_ptr() == value.data_ptr()
                    and other.shape == value.shape == result.shape
                ):
                    return self.linear(x, w, bias, other)
        return None

    def __torch_function__(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(function, "__name__", None)
        if name in ("sin", "cos", "exp", "tanh"):
            return self.scalar(name, args[0])
        if name == "__rpow__":
            return self.scalar(
                "exp", args[0] * torch.tensor(math.log(args[1]), dtype=args[0].dtype)
            )
        if name == "rsqrt":
            return torch.ones_like(args[0]) / torch.sqrt(args[0])
        if (
            self.whole
            and name in ("add", "__add__", "__radd__")
            and len(args) == 2
            and not kwargs
        ):
            fused = self.residual(args[0], args[1])
            del self.linears[:-8]
            if fused is not None:
                return fused
        if name == "silu":
            return args[0] / (1 + operator_exp(-args[0]))
        if name == "mean":
            return mean(*args, **kwargs)
        if name == "softmax":
            dim = args[1] if len(args) > 1 else kwargs["dim"]
            return ball_softmax(args[0], dim)
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
            return (values * 0.5) * (1 + operator_tanh(argument))
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
