from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from framework.quant.core.bwq import BwqPackage, validate_bwq

DA_ADDR = 0
DW_BASE_ADDR = 16
DMA_BYTES = 16


@dataclass(frozen=True)
class WeightRuntimeSpec:
    name: str
    dw_addr: int
    per_channel: bool


def weight_runtime_specs(pkg: BwqPackage) -> list[WeightRuntimeSpec]:
    validate_bwq(pkg)
    specs = []
    for tensor in pkg.tensors:
        if tensor.storage != "i8":
            raise ValueError(f"runtime only supports i8 weights: {tensor.name}")
        if tensor.axes not in ([], [0]):
            raise ValueError(f"runtime only supports per-tensor or output-channel scales: {tensor.name}")
        if tensor.scale_off % DMA_BYTES:
            raise ValueError(f"unaligned scale offset: {tensor.name}")
        specs.append(WeightRuntimeSpec(tensor.name, DW_BASE_ADDR + tensor.scale_off,
                                       bool(tensor.axes)))
    if DW_BASE_ADDR + len(pkg.scales) > 5120:
        raise ValueError("scale image exceeds Pebble MMIO capacity")
    return specs


def emit_runtime_header(pkg: BwqPackage, path: Path | str) -> list[WeightRuntimeSpec]:
    specs = weight_runtime_specs(pkg)
    path = Path(path)
    image = ", ".join(f"0x{byte:02x}" for byte in pkg.scales)
    lines = [
        "#ifndef BBWQ_RUNTIME_H", "#define BBWQ_RUNTIME_H", "",
        "#include <stdint.h>", "#include <bbhw/isa/isa.h>", "",
        f"#define BBWQ_DA_ADDR {DA_ADDR}",
        f"#define BBWQ_DW_BASE_ADDR {DW_BASE_ADDR}",
        f"#define BBWQ_SCALE_IMAGE_BYTES {len(pkg.scales)}", "",
        f"static const uint8_t bbwq_scale_image[{len(pkg.scales)}] = {{{image}}};", "",
        "static inline void bbwq_load_scales(void) {",
        "  bb_mvin_mmio((uintptr_t)bbwq_scale_image, BBWQ_DW_BASE_ADDR,",
        "               BBWQ_SCALE_IMAGE_BYTES / 16, 16);", "}", "",
    ]
    for spec in specs:
        name = re.sub("[^A-Za-z0-9]", "_", spec.name).upper()
        lines.extend([
            f"#define BBWQ_{name}_DW_ADDR {spec.dw_addr}",
            f"#define BBWQ_{name}_PER_CHANNEL {int(spec.per_channel)}", "",
        ])
    lines.append("#endif")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return specs
