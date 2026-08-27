from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
import subprocess

from framework.quant.core.bwq import BwqPackage, validate_bwq

DA_ADDR = 0
DW_BASE_ADDR = 16
MMIO_BYTES = 5120
DMA_BYTES = 16
SCALE_LANES = 16


def scale_count(shape: list[int], axes: list[int]) -> int:
    if axes == [0]:
        return ((shape[0] + SCALE_LANES - 1) // SCALE_LANES) * SCALE_LANES
    if not axes:
        return 1
    raise ValueError(f"unsupported scale axes: {axes}")


@dataclass
class QuantTensor:
    name: str
    shape: list[int]
    payload_shape: list[int]
    storage: str
    axes: list[int]
    payload_off: int
    payload_len: int
    scale_off: int
    scale_len: int


@dataclass
class RaxQuantPackage:
    tensors: list[QuantTensor]
    weights_i8: bytes
    params_f32: bytes
    scales_f32: bytes
    assets: dict[str, bytes]


def rax_from_bwq(pkg: BwqPackage) -> RaxQuantPackage:
    validate_bwq(pkg)
    tensors: list[QuantTensor] = []
    weights: list[bytes] = []
    scales: list[bytes] = []
    weight_off = scale_off = 0
    for source in pkg.tensors:
        if source.storage != "i8" or source.axes not in ([], [0]):
            raise ValueError(f"RAX W8A8 requires i8 tensor/channel weight: {source.name}")
        weight = pkg.weights[source.weight_off : source.weight_off + source.weight_len]
        raw_scales = pkg.scales[source.scale_off : source.scale_off + source.scale_len]
        raw_count = 1 if not source.axes else source.shape[0]
        values = struct.unpack(f"<{raw_count}f", raw_scales)
        padded_count = scale_count(source.shape, source.axes)
        padded = values + (1.0,) * (padded_count - raw_count)
        scale = struct.pack(f"<{padded_count}f", *padded)
        tensors.append(QuantTensor(source.name, source.shape, source.shape,
                                   source.storage, source.axes, weight_off,
                                   len(weight), scale_off, len(scale)))
        weights.append(weight)
        scales.append(scale)
        weight_off += len(weight)
        scale_off += len(scale)
    return RaxQuantPackage(tensors, b"".join(weights), b"", b"".join(scales), {})


def _numel(shape: list[int], name: str) -> int:
    n = 1
    for d in shape:
        if d <= 0:
            raise ValueError(f"bad shape for {name}")
        n *= d
    return n


def validate_rax_quant(pkg: RaxQuantPackage) -> None:
    if not pkg.tensors:
        raise ValueError("empty tensor list")
    names: set[str] = set()
    weight_end = param_end = scale_end = 0
    for tensor in pkg.tensors:
        if tensor.name in names:
            raise ValueError(f"duplicate tensor name: {tensor.name}")
        names.add(tensor.name)
        if tensor.storage not in ("i8", "f32"):
            raise ValueError(f"unsupported storage for {tensor.name}: {tensor.storage}")
        if tensor.payload_off < 0 or tensor.payload_len < 0:
            raise ValueError(f"bad payload range for {tensor.name}")
        if tensor.scale_off < 0 or tensor.scale_len < 0:
            raise ValueError(f"bad scale range for {tensor.name}")
        numel = _numel(tensor.shape, tensor.name)
        if _numel(tensor.payload_shape, tensor.name) != numel:
            raise ValueError(f"payload shape mismatch for {tensor.name}")
        if tensor.storage == "i8":
            if tensor.axes not in ([], [0]):
                raise ValueError(f"unsupported scale axes for {tensor.name}: {tensor.axes}")
            if tensor.payload_off + tensor.payload_len > len(pkg.weights_i8):
                raise ValueError(f"weight OOB for {tensor.name}")
            if tensor.payload_off != weight_end:
                raise ValueError(f"non-contiguous weight payload for {tensor.name}")
            if tensor.payload_len != numel:
                raise ValueError(f"weight_len mismatch for {tensor.name}")
            if tensor.scale_off + tensor.scale_len > len(pkg.scales_f32):
                raise ValueError(f"scale OOB for {tensor.name}")
            scale_n = scale_count(tensor.shape, tensor.axes)
            if tensor.scale_len != scale_n * 4:
                raise ValueError(f"scale_len mismatch for {tensor.name}")
            if tensor.scale_off % 4:
                raise ValueError(f"unaligned scale offset for {tensor.name}")
            if tensor.scale_off != scale_end:
                raise ValueError(f"non-contiguous scale payload for {tensor.name}")
            scales = struct.unpack_from(f"<{scale_n}f", pkg.scales_f32, tensor.scale_off)
            if any(not math.isfinite(scale) or scale <= 0.0 for scale in scales):
                raise ValueError(f"invalid scale value for {tensor.name}")
            if tensor.axes == [0] and any(scale != 1.0 for scale in scales[tensor.shape[0] :]):
                raise ValueError(f"bad channel-scale padding for {tensor.name}")
            weight_end += tensor.payload_len
            scale_end += tensor.scale_len
        else:
            if tensor.axes or tensor.scale_off or tensor.scale_len:
                raise ValueError(f"f32 parameter has quant metadata: {tensor.name}")
            if tensor.payload_off + tensor.payload_len > len(pkg.params_f32):
                raise ValueError(f"f32 parameter OOB for {tensor.name}")
            if tensor.payload_off != param_end:
                raise ValueError(f"non-contiguous f32 payload for {tensor.name}")
            if tensor.payload_len != numel * 4:
                raise ValueError(f"f32 parameter size mismatch for {tensor.name}")
            param_end += tensor.payload_len
    if weight_end != len(pkg.weights_i8):
        raise ValueError("unused INT8 weight payload")
    if param_end != len(pkg.params_f32):
        raise ValueError("unused FP32 parameter payload")
    if scale_end != len(pkg.scales_f32):
        raise ValueError("unused scale payload")


def _payload_dir(rax: Path) -> Path:
    return rax.parent / f"{rax.stem}.payload"


def _scale_image(pkg: RaxQuantPackage) -> bytes:
    image = pkg.scales_f32 + bytes((-len(pkg.scales_f32)) % DMA_BYTES)
    if DW_BASE_ADDR + len(image) > MMIO_BYTES:
        raise ValueError("scale image exceeds Pebble MMIO capacity")
    return image


def _quant_index(pkg: RaxQuantPackage) -> dict:
    tensors = []
    for tensor in pkg.tensors:
        tensors.append({
            "name": tensor.name,
            "shape": tensor.shape,
            "payload_shape": tensor.payload_shape,
            "storage": tensor.storage,
            "payload": "weights_i8" if tensor.storage == "i8" else "params_f32",
            "payload_offset": tensor.payload_off,
            "payload_bytes": tensor.payload_len,
            "dw_addr": DW_BASE_ADDR + tensor.scale_off if tensor.storage == "i8" else 0,
            "dw_bytes": tensor.scale_len,
            "per_channel": bool(tensor.axes) if tensor.storage == "i8" else False,
        })
    return {"version": 1, "da_addr": DA_ADDR, "dw_base_addr": DW_BASE_ADDR, "tensors": tensors}


def _manifest(model_name: str, payload_name: str, weights_bytes: int, params_bytes: int,
              scales_bytes: int, index_bytes: int, assets: dict[str, bytes]) -> str:
    asset_constants = "".join(
        f'  rhal.constant @{name.replace(".", "_")} {{id = {5 + i} : i32, storage = "external",\n'
        f'                             type = tensor<{len(data)}xi8>,\n'
        f'                             uri = "file:{payload_name}/{name}"}}\n'
        for i, (name, data) in enumerate(assets.items())
    )
    return f'''rhal.module @quant_{model_name} attributes {{
    version = "0.1.0",
    model_name = "{model_name}"}} {{
  rhal.constant @weights_i8 {{id = 1 : i32, storage = "external",
                             type = tensor<{weights_bytes}xi8>,
                             uri = "file:{payload_name}/weights.i8"}}
  rhal.constant @params_f32 {{id = 2 : i32, storage = "external",
                             type = tensor<{params_bytes}xi8>,
                             uri = "file:{payload_name}/params.f32"}}
  rhal.constant @scales_f32 {{id = 3 : i32, storage = "external",
                              type = tensor<{scales_bytes}xi8>,
                              uri = "file:{payload_name}/scales.bin"}}
  rhal.constant @quant_index {{id = 4 : i32, storage = "external",
                               type = tensor<{index_bytes}xi8>,
                               uri = "file:{payload_name}/quant-index.json"}}
{asset_constants}}}
'''


def write_rax(pkg: RaxQuantPackage, rax: Path | str, rax_pack: Path | str,
              model_name: str) -> None:
    validate_rax_quant(pkg)
    rax = Path(rax)
    rax_pack = Path(rax_pack)
    if not rax_pack.is_file():
        raise ValueError(f"rax-pack not found: {rax_pack}")
    payload_dir = _payload_dir(rax)
    payload_dir.mkdir(parents=True, exist_ok=True)
    scales = _scale_image(pkg)
    index = json.dumps(_quant_index(pkg), sort_keys=True, indent=2).encode("ascii")
    (payload_dir / "weights.i8").write_bytes(pkg.weights_i8)
    (payload_dir / "params.f32").write_bytes(pkg.params_f32)
    (payload_dir / "scales.bin").write_bytes(scales)
    (payload_dir / "quant-index.json").write_bytes(index)
    for name, data in pkg.assets.items():
        (payload_dir / name).write_bytes(data)
    manifest = payload_dir / "quant.rhal.mlir"
    manifest.write_text(_manifest(model_name, payload_dir.name, len(pkg.weights_i8),
                                  len(pkg.params_f32), len(scales), len(index), pkg.assets),
                        encoding="ascii")
    subprocess.run([str(rax_pack), str(manifest), "-o", str(rax), "--embed-payload"], check=True)
