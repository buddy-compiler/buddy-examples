"""MXFP quant/dequant. E4M3/E2M1 use pure-NumPy LUT (no ml_dtypes) per env constraint; OCP-faithful nearest."""

from __future__ import annotations

from typing import Literal

import numpy as np

BLOCK = 32
_E4M3_MAX = 448.0
_E2M1_MAX = 6.0
_E8M0_NAN = 0xFF


def _build_e4m3_lut() -> np.ndarray:
    table = np.zeros(256, dtype=np.float32)
    for byte in range(256):
        s = (byte >> 7) & 1
        e = (byte >> 3) & 0xF
        m = byte & 0x7
        if e == 15 and m == 7:
            table[byte] = np.nan
        elif e == 0:
            val = (m / 8.0) * (2.0**-6)
            table[byte] = -val if s else val
        else:
            val = (1.0 + m / 8.0) * (2.0 ** (e - 7))
            table[byte] = -val if s else val
    return table


def _build_e2m1_lut() -> np.ndarray:
    table = np.zeros(16, dtype=np.float32)
    for nibble in range(16):
        s = (nibble >> 3) & 1
        e = (nibble >> 1) & 0x3
        m = nibble & 1
        if e == 0:
            val = (m / 2.0) * (2.0**0)
            table[nibble] = -val if s else val
        else:
            val = (1.0 + m / 2.0) * (2.0 ** (e - 1))
            table[nibble] = -val if s else val
    return table


_E4M3_LUT = _build_e4m3_lut()
_E2M1_LUT = _build_e2m1_lut()


def _require_finite(x: np.ndarray, what: str = "input") -> None:
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{what} must be finite")


def _encode_via_lut(x: np.ndarray, lut: np.ndarray, n: int) -> np.ndarray:
    flat = np.asarray(x, dtype=np.float32).ravel()
    diffs = np.abs(flat[:, None] - lut[None, :n])
    finite = np.isfinite(lut[:n])
    diffs = np.where(finite[None, :], diffs, np.inf)
    return np.argmin(diffs, axis=1).astype(np.uint8)


def _encode_e4m3(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    _require_finite(arr, "e4m3 encode input")
    sat = np.clip(arr, -_E4M3_MAX, _E4M3_MAX)
    return _encode_via_lut(sat, _E4M3_LUT, 256).reshape(arr.shape)


def _encode_e2m1(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    _require_finite(arr, "e2m1 encode input")
    sat = np.clip(arr, -_E2M1_MAX, _E2M1_MAX)
    return _encode_via_lut(sat, _E2M1_LUT, 16).reshape(arr.shape)


def _decode_e4m3(b: np.ndarray) -> np.ndarray:
    return _E4M3_LUT[np.asarray(b, dtype=np.uint8)]


def _decode_e2m1(b: np.ndarray) -> np.ndarray:
    return _E2M1_LUT[np.asarray(b, dtype=np.uint8) & 0xF]


def _float_to_e8m0(x: float) -> int:
    if not np.isfinite(x):
        raise ValueError("e8m0 input must be finite")
    if x <= 0.0:
        raise ValueError("e8m0 input must be positive")
    f = np.float32(x)
    bits = f.view(np.uint32)
    exp = int((bits >> 23) & 0xFF)
    mant = bits & 0x7FFFFF
    if exp == 0:
        if mant == 0:
            return 0
        e = int(np.ceil(np.log2(float(f))))
        return e + 127
    if mant != 0:
        exp += 1
    if exp >= _E8M0_NAN:
        return _E8M0_NAN - 1
    return exp


def _e8m0_to_scale(e: int) -> np.float32:
    if e == _E8M0_NAN:
        raise ValueError("e8m0 NaN scale")
    return np.float32(2.0 ** (e - 127))


def _fmt_max(fmt: str) -> float:
    if fmt == "mxfp8":
        return _E4M3_MAX
    if fmt == "mxfp4":
        return _E2M1_MAX
    raise ValueError(f"unsupported fmt: {fmt}")


def _blocks(w: np.ndarray) -> np.ndarray:
    if w.size % BLOCK != 0:
        raise ValueError(f"numel {w.size} not divisible by {BLOCK}")
    return np.ascontiguousarray(w, dtype=np.float32).reshape(-1, BLOCK)


def _pack_nibbles(codes: np.ndarray) -> bytes:
    flat = np.asarray(codes, dtype=np.uint8).ravel()
    if flat.size % 2 != 0:
        raise ValueError("nibble count must be even")
    out = (flat[1::2] << 4) | (flat[0::2] & 0xF)
    return out.astype(np.uint8).tobytes()


def _unpack_nibbles(packed: bytes, n: int) -> np.ndarray:
    raw = np.frombuffer(packed, dtype=np.uint8)
    need = n // 2
    if raw.size < need:
        raise ValueError(f"packed len {raw.size} < {need}")
    out = np.empty(n, dtype=np.uint8)
    out[0::2] = raw[:need] & 0xF
    out[1::2] = (raw[:need] >> 4) & 0xF
    return out


def quantize_mxfp(
    w: np.ndarray, fmt: Literal["mxfp4", "mxfp8"]
) -> tuple[bytes, np.ndarray]:
    w = np.asarray(w, dtype=np.float32)
    _require_finite(w, "weight")
    blk = _blocks(w)
    nblk = blk.shape[0]
    scales = np.empty(nblk, dtype=np.uint8)
    elems_max = _fmt_max(fmt)
    encoded_blocks: list[np.ndarray] = []

    for i in range(nblk):
        block = blk[i]
        amax = float(np.max(np.abs(block)))
        if amax == 0.0:
            raise ValueError("block maxAbs is 0")
        e8 = _float_to_e8m0(amax / elems_max)
        scales[i] = e8
        scale = _e8m0_to_scale(e8)
        scaled = block / scale
        if fmt == "mxfp8":
            encoded_blocks.append(_encode_e4m3(scaled))
        else:
            encoded_blocks.append(_encode_e2m1(scaled))

    codes = np.concatenate(encoded_blocks)
    if fmt == "mxfp8":
        packed = codes.astype(np.uint8).tobytes()
    else:
        packed = _pack_nibbles(codes)
    return packed, scales


def dequantize_mxfp(
    packed: bytes, scales: np.ndarray, numel: int, fmt: str
) -> np.ndarray:
    if numel % BLOCK != 0:
        raise ValueError(f"numel {numel} not divisible by {BLOCK}")
    nblk = numel // BLOCK
    scales = np.asarray(scales, dtype=np.uint8).reshape(-1)
    if scales.size != nblk:
        raise ValueError(f"scales len {scales.size} != {nblk}")

    if fmt == "mxfp8":
        if len(packed) != numel:
            raise ValueError(f"packed len {len(packed)} != numel {numel}")
        codes = np.frombuffer(packed, dtype=np.uint8)
        dec = _decode_e4m3(codes)
    elif fmt == "mxfp4":
        need = numel // 2
        if len(packed) != need:
            raise ValueError(f"packed len {len(packed)} != {need}")
        codes = _unpack_nibbles(packed, numel)
        dec = _decode_e2m1(codes)
    else:
        raise ValueError(f"unsupported fmt: {fmt}")

    out = np.empty(numel, dtype=np.float32)
    for i in range(nblk):
        s = _e8m0_to_scale(int(scales[i]))
        sl = slice(i * BLOCK, (i + 1) * BLOCK)
        out[sl] = dec[sl] * s
    return out.astype(np.float16)
