from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from framework.quant.core.rax import DA_ADDR, DW_BASE_ADDR, MMIO_BYTES, scale_count


@dataclass(frozen=True)
class RaxWeightBinding:
    dw_addr: int
    dw_bytes: int
    per_channel: bool


def load_rax_weight_bindings(index_path: Path | str) -> dict[str, RaxWeightBinding]:
    index = json.loads(Path(index_path).read_text(encoding="ascii"))
    if not isinstance(index, dict) or index.get("version") != 1:
        raise ValueError("unsupported quant-index version")
    if index.get("da_addr") != DA_ADDR or index.get("dw_base_addr") != DW_BASE_ADDR:
        raise ValueError("quant-index MMIO layout mismatch")
    tensors = index.get("tensors")
    if not isinstance(tensors, list):
        raise ValueError("quant-index tensors must be a list")
    bindings: dict[str, RaxWeightBinding] = {}
    expected_dw_addr = DW_BASE_ADDR
    for tensor in tensors:
        if not isinstance(tensor, dict):
            raise ValueError("quant-index tensor must be an object")
        if tensor.get("storage") != "i8":
            continue
        name = tensor.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("quant-index INT8 weight has invalid name")
        if name in bindings:
            raise ValueError(f"duplicate quant binding: {name}")
        dw_addr = tensor.get("dw_addr")
        dw_bytes = tensor.get("dw_bytes")
        per_channel = tensor.get("per_channel")
        shape = tensor.get("shape")
        if type(dw_addr) is not int or type(dw_bytes) is not int:
            raise ValueError(f"invalid Dw layout for {name}")
        if not isinstance(per_channel, bool):
            raise ValueError(f"invalid per_channel flag for {name}")
        if not isinstance(shape, list) or not shape or any(type(dim) is not int or dim <= 0 for dim in shape):
            raise ValueError(f"invalid shape for {name}")
        scale_n = scale_count(shape, [0] if per_channel else [])
        if dw_addr != expected_dw_addr or dw_addr % 4:
            raise ValueError(f"unaligned Dw address for {name}")
        if dw_bytes != scale_n * 4 or dw_addr + dw_bytes > MMIO_BYTES:
            raise ValueError(f"bad Dw layout for {name}")
        bindings[name] = RaxWeightBinding(dw_addr, dw_bytes, per_channel)
        expected_dw_addr += dw_bytes
    if not bindings:
        raise ValueError("quant-index has no INT8 weights")
    return bindings


def bind_rax_quant_metadata(graph, parameter_names: dict[str, str],
                            bindings: dict[str, RaxWeightBinding]) -> None:
    weight_arg_index = {
        "Conv2dOp": 1, "QuantizedConv2dOp": 1,
        "MatmulOp": 1, "QuantizedMatmulOp": 1,
        "AddMMOp": 2, "QuantizedAddMMOp": 2,
    }
    for node in graph._body:
        arg_index = weight_arg_index.get(type(node).__name__)
        if arg_index is None:
            continue
        parameter = str(node.args[arg_index])
        while parameter not in parameter_names:
            source = graph.node_table[parameter]
            if len(source._parents) != 1:
                raise ValueError(f"missing source weight name for {node.name}")
            parameter = source._parents[0]
        weight_name = parameter_names[parameter]
        if weight_name not in bindings:
            raise ValueError(f"missing RAX Dw metadata for {weight_name}")
        binding = bindings[weight_name]
        node._weight_name = weight_name
        node._dw_addr = binding.dw_addr
        node._dw_bytes = binding.dw_bytes
        node._per_channel = binding.per_channel
