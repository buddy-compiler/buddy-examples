from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np

from quant.core.bwq import BwqPackage, TensorSpec, read_bwq, write_bwq
from quant.core.metrics import nmse
from quant.core.mxfp import dequantize_mxfp, quantize_mxfp
from quant.core.quantize import quantize_symmetric

_STORAGE = {"i8", "mxfp4", "mxfp8"}
_DTYPE = {"f32", "bf16"}


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    u16 = np.asarray(bits, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)


def _f32_to_bf16(arr: np.ndarray) -> np.ndarray:
    u32 = np.asarray(arr, dtype=np.float32).view(np.uint32)
    return (u32 >> 16).astype(np.uint16)


def _load_mode(path: Path) -> dict:
    meta = tomllib.loads(path.read_bytes().decode("utf-8"))
    if "storage" not in meta:
        raise ValueError("mode missing storage")
    storage = meta["storage"]
    if storage not in _STORAGE:
        raise ValueError(f"unsupported mode storage: {storage!r}")
    default_axes = list(meta.get("default_axes", []))
    return {"storage": storage, "default_axes": default_axes}


def _load_shapes(path: Path) -> list[dict]:
    meta = tomllib.loads(path.read_bytes().decode("utf-8"))
    tensors = meta.get("tensor")
    if not tensors:
        raise ValueError("shapes: empty tensor list")
    out: list[dict] = []
    for entry in tensors:
        if "name" not in entry or "shape" not in entry:
            raise ValueError("shapes entry missing name or shape")
        out.append(entry)
    return out


def _load_arg0(path: Path, dtype: str) -> np.ndarray:
    raw = path.read_bytes()
    if dtype == "f32":
        if len(raw) % 4 != 0:
            raise ValueError(f"arg0 size {len(raw)} not multiple of 4")
        return np.frombuffer(raw, dtype=np.float32).copy()
    if dtype == "bf16":
        if len(raw) % 2 != 0:
            raise ValueError(f"arg0 size {len(raw)} not multiple of 2")
        return _bf16_to_f32(np.frombuffer(raw, dtype=np.uint16))
    raise ValueError(f"unsupported dtype: {dtype!r}")


def _write_recon(path: Path, arr: np.ndarray, dtype: str) -> None:
    if dtype == "f32":
        path.write_bytes(np.asarray(arr, dtype=np.float32).tobytes())
    elif dtype == "bf16":
        path.write_bytes(_f32_to_bf16(arr).tobytes())
    else:
        raise ValueError(f"unsupported dtype: {dtype!r}")


def _dequant_i8(q: np.ndarray, scale: np.ndarray, axes: list[int]) -> np.ndarray:
    if not axes:
        return q.astype(np.float32) / float(np.asarray(scale).reshape(()))
    shape = [1] * q.ndim
    for ax in axes:
        shape[ax] = q.shape[ax]
    return q.astype(np.float32) / scale.reshape(shape).astype(np.float32)


def _to_dtype(arr: np.ndarray, dtype: str) -> np.ndarray:
    f32 = np.asarray(arr, dtype=np.float32)
    if dtype == "f32":
        return f32
    if dtype == "bf16":
        return _bf16_to_f32(_f32_to_bf16(f32))
    raise ValueError(f"unsupported dtype: {dtype!r}")


def pack(
    arg0: Path | str,
    shapes: Path | str,
    mode: Path | str,
    out_bwq: Path | str,
    out_recon: Path | str,
    *,
    dtype: str = "f32",
) -> float:
    if dtype not in _DTYPE:
        raise ValueError(f"unsupported dtype: {dtype!r}")

    arg0 = Path(arg0)
    shapes = Path(shapes)
    mode = Path(mode)
    out_bwq = Path(out_bwq)
    out_recon = Path(out_recon)

    cfg = _load_mode(mode)
    entries = _load_shapes(shapes)
    blob = _load_arg0(arg0, dtype)
    ref_f32 = blob.astype(np.float32)

    cursor = 0
    w_parts: list[bytes] = []
    s_parts: list[bytes] = []
    specs: list[TensorSpec] = []
    recon_parts: list[np.ndarray] = []
    w_off = s_off = 0

    for entry in entries:
        name = entry["name"]
        shape = list(entry["shape"])
        n = int(np.prod(shape))
        if cursor + n > blob.size:
            raise ValueError(f"arg0 truncated at {name}")
        w = blob[cursor : cursor + n].reshape(shape).astype(np.float32)
        cursor += n

        storage = entry.get("storage", cfg["storage"])
        if storage not in _STORAGE:
            raise ValueError(f"unsupported storage {storage!r} for {name}")
        if storage in ("mxfp4", "mxfp8"):
            axes: list[int] = []
        else:
            axes = list(entry.get("axes", cfg["default_axes"]))

        if storage == "i8":
            q, scale = quantize_symmetric(w, axes)
            wb = q.tobytes()
            sb = np.asarray(scale, dtype=np.float32).tobytes()
            recon = _dequant_i8(q, scale, axes)
        else:
            wb, scale = quantize_mxfp(w, storage)
            sb = np.asarray(scale, dtype=np.uint8).tobytes()
            recon = dequantize_mxfp(wb, scale, n, storage).astype(np.float32)

        specs.append(
            TensorSpec(name, shape, storage, axes, w_off, len(wb), s_off, len(sb))
        )
        w_parts.append(wb)
        s_parts.append(sb)
        w_off += len(wb)
        s_off += len(sb)
        recon_parts.append(_to_dtype(recon, dtype).reshape(-1))

    if cursor != blob.size:
        raise ValueError(f"arg0 leftover {blob.size - cursor} elements")

    write_bwq(
        BwqPackage(1, specs, b"".join(w_parts), b"".join(s_parts)),
        out_bwq,
    )
    recon_all = np.concatenate(recon_parts)
    _write_recon(out_recon, recon_all, dtype)
    read_bwq(out_bwq)
    return nmse(ref_f32, _load_arg0(out_recon, dtype))
