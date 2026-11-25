"""The exp, the erf of GELU and the softmax of the goban thread operators in PyTorch,
operation by operation as GobanDefaultConfig/models/*/operators.mlir has them
(mlp-mixer operator_3 for erf; the exp and reduction operators for softmax).

erf is MLIR's port of the C library's erff: a rational function of x^2 below
0.84375, of |x| - 1 below 1.25, and exp(-z^2 - 0.5625) exp((z - |x|)(z + |x|)
+ R/S) / |x| up to 6, with exp in FP64 from a table of 32 entries; every
polynomial in fused multiply-adds.
"""

import torch


def f32(value):
    return torch.tensor(value, dtype=torch.float32)


def fma(a, b, c):
    """a * b + c rounded once to FP32: the product is exact in FP64, and the
    sum with c, rounded to odd in FP64, rounds to FP32 as the exact sum."""
    a, b, c = torch.broadcast_tensors(
        *(torch.as_tensor(v, dtype=torch.float32) for v in (a, b, c))
    )
    product = a.double() * b.double()
    addend = c.double()
    total = product + addend
    part = total - product
    error = (product - (total - part)) + (addend - part)
    even = (total.contiguous().view(torch.int64) & 1) == 0
    toward = torch.where(error > 0, torch.inf, -torch.inf).to(torch.float64)
    return torch.where(
        (error != 0) & even, torch.nextafter(total, toward), total
    ).float()


def fma64(a, b, c):
    """a * b + c in FP64 from the exact product (Dekker) and the exact sum
    (Knuth): the fused result but for a double rounding about once in 2^50."""
    a, b, c = torch.broadcast_tensors(
        *(torch.as_tensor(v, dtype=torch.float64) for v in (a, b, c))
    )
    product = a * b
    split = 134217729.0
    a_high = a * split
    a_high = a_high - (a_high - a)
    a_low = a - a_high
    b_high = b * split
    b_high = b_high - (b_high - b)
    b_low = b - b_high
    low = (
        (a_high * b_high - product) + a_high * b_low + a_low * b_high
    ) + a_low * b_low
    total = product + c
    part = total - product
    error = (product - (total - part)) + (c - part)
    return total + (error + low)


def horner(x, coefficients):
    """fma(x, fma(x, c0, c1), c2) …: the coefficients from the highest."""
    value = fma(x, f32(coefficients[0]), f32(coefficients[1]))
    for coefficient in coefficients[2:]:
        value = fma(x, value, f32(coefficient))
    return value


exp2f_table = torch.tensor(
    [
        4607182418800017408,
        4607140297302181236,
        4607100335213349135,
        4607062579818421073,
        4607027079437701499,
        4606993883449571754,
        4606963042313658936,
        4606934607594512097,
        4606908631985796885,
        4606885169335019979,
        4606864274668794914,
        4606846004218661165,
        4606830415447468583,
        4606817567076339586,
        4606807519112221737,
        4606800332876043653,
        4606796071031487437,
        4606794797614391156,
        4606796578062795143,
        4606801479247646227,
        4606809569504174299,
        4606820918663955941,
        4606835598087680144,
        4606853680698631517,
        4606875241016906669,
        4606900355194379847,
        4606929101050434204,
        4606961558108475497,
        4606997807633245319,
        4607037932668951391,
        4607082018078232794,
        4607130150581978432,
    ],
    dtype=torch.int64,
)


def exp_table(x):
    """The exp of erf's tail: FP64 with k = round(32 x / ln 2) through the
    shift 1.5 * 2^52, the table of 2^(i/32) and a cubic, truncated to FP32."""
    x = x.double()
    scale, shift = 46.166241308446828, 6755399441055744.0
    k = fma64(scale, x, shift)
    bits = k.contiguous().view(torch.int64)
    r = fma64(scale, x, -(k - shift))
    power = (exp2f_table[bits & 31] + (bits << 47)).view(torch.float64)
    p = fma64(1.6938359250920212e-6, r, 2.3459809789509004e-4)
    q = fma64(0.021660849396613134, r, 1.0)
    return (fma64(p, r * r, q) * power).float()


def erf(x):
    bits = x.contiguous().view(torch.int32)
    magnitude = bits & 0x7FFFFFFF
    absolute = magnitude.view(torch.float32)
    negative = bits < 0

    # |x| < 0.84375
    z = x * x
    r = horner(
        z, (-2.37630175e-5, -0.00577027025, -0.0284817498, -0.325042099, 0.128379166)
    )
    s = horner(
        z,
        (-3.96022824e-6, 1.32494737e-4, 0.00508130621, 0.0650222525, 0.397917211, 1.0),
    )
    small = torch.where(
        magnitude < 0x04000000,
        fma(f32(1.02703333), x, x * f32(8.0)) * f32(0.125),
        fma(f32(0.128379166), x, x),
    )
    near = torch.where(magnitude < 0x31800000, small, fma(x, r / s, x))

    # 0.84375 <= |x| < 1.25
    s = absolute - f32(1.0)
    p = horner(
        s,
        (
            -0.00216637552,
            0.0354783051,
            -0.110894695,
            0.31834662,
            -0.37220788,
            0.414856106,
            -0.00236211857,
        ),
    )
    q = horner(
        s,
        (
            1.198450e-02,
            0.0136370836,
            0.126171216,
            0.0718286559,
            0.540397942,
            0.106420882,
            1.0,
        ),
    )
    middle = p / q + f32(0.845062911)
    middle = torch.where(negative, -middle, middle)

    # 1.25 <= |x| < 6
    s = f32(1.0) / (absolute * absolute)
    ra = horner(
        s,
        (
            -9.81432914,
            -81.2874374,
            -184.605087,
            -162.396667,
            -62.3753319,
            -10.5586262,
            -0.693858563,
            -0.00986494403,
        ),
    )
    sa = horner(
        s,
        (
            -0.0604244135,
            6.57024956,
            1.086350e02,
            429.008148,
            645.387268,
            434.565887,
            137.657761,
            19.6512718,
            1.0,
        ),
    )
    rb = horner(
        s,
        (
            -483.519196,
            -1025.09509,
            -637.566467,
            -160.636383,
            -17.7579556,
            -0.799283266,
            -0.0098649431,
        ),
    )
    sb = horner(
        s,
        (
            -22.4409523,
            474.528534,
            2553.05029,
            3199.85815,
            1536.72961,
            325.792511,
            30.3380604,
            1.0,
        ),
    )
    quotient = torch.where(magnitude < 1077336942, ra / sa, rb / sb)
    z = (magnitude & -4096).view(torch.float32)
    tail = exp_table(fma(-z, z, f32(-0.5625))) * exp_table(
        fma(z - absolute, z + absolute, quotient)
    )
    tail = tail / absolute
    far = torch.where(negative, tail - f32(1.0), f32(1.0) - tail)

    # 6 <= |x|, and what is not finite
    one = torch.where(negative, f32(-1.0), f32(1.0))
    return torch.where(
        magnitude < 0x3F580000,
        near,
        torch.where(
            magnitude < 0x3FA00000,
            middle,
            torch.where(
                magnitude < 0x40C00000,
                far,
                torch.where(magnitude < 0x7F800000, one, one + f32(1.0) / x),
            ),
        ),
    )


def gelu(x):
    """GELU as the operators compute it: (x * 0.5) * (erf(x * 0.707106769) + 1)."""
    return (x * f32(0.5)) * (erf(x * f32(0.707106769)) + f32(1.0))


def exp(x):
    """math.exp of the operators (MLIR's Cephes exp): x clamped to
    [-87.8, 88.8], k = floor(x log2(e) + 0.5) within +-127, a polynomial of
    x - k ln 2 (ln 2 in two terms), times 2^k (zero at -127)."""
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


def softmax(value, dim):
    """exp(x - max) over its sum, the sum in order from zero as the
    operators' reduction adds it, then one division an element."""
    weights = exp(value - value.amax(dim=dim, keepdim=True))
    parts = weights.movedim(dim, 0)
    total = torch.zeros_like(parts[0])
    for part in parts:
        total = total + part
    return weights / total.unsqueeze(dim)
