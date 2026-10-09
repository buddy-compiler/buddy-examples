import math

import torch
from torch import nn

from stack.compiler.quant import mxfp8_cache as codec


@torch.library.custom_op("gemma::panel_linear", mutates_args=())
def panel_linear(
    values: torch.Tensor,
    packed: torch.Tensor,
    columns: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    bank_bytes: int,
) -> torch.Tensor:
    # Capture reference: decode exactly the original panels, retaining the full K.
    reduction = values.shape[1]
    chunks = math.ceil(reduction / tile_k)
    panels = []
    for first in range(0, columns, tile_n):
        parts = []
        for part, start in enumerate(range(0, reduction, tile_k)):
            width = min(tile_k, reduction - start)
            offset = ((first // tile_n) * chunks + part) * bank_bytes
            payload = packed[offset : offset + bank_bytes]
            codes = payload[: tile_n * width].reshape(tile_n, width)
            scales = payload[tile_n * width : tile_n * width * 33 // 32].reshape(
                tile_n, width // 32
            )
            parts.append(codec.decode(codes, scales))
        panels.append(torch.cat(parts, dim=1)[: min(tile_n, columns - first)])
    codes, scales = codec.encode(values)
    activation = codec.decode(codes, scales)
    return torch.cat(
        [nn.functional.linear(activation, panel) for panel in panels], dim=1
    )


@panel_linear.register_fake
def _panel_linear_fake(values, packed, columns, tile_m, tile_n, tile_k, bank_bytes):
    if (
        values.ndim != 2
        or values.dtype != torch.float32
        or packed.ndim != 1
        or packed.dtype != torch.int8
        or values.shape[1] % 32
        or columns <= 0
        or tile_n <= 0
        or tile_n % 16
        or tile_k <= 0
        or tile_k % 32
        or packed.numel()
        != math.ceil(columns / tile_n)
        * math.ceil(values.shape[1] / tile_k)
        * bank_bytes
    ):
        raise ValueError(
            "Gemma panel task does not match its original packed weight layout"
        )
    return values.new_empty((values.shape[0], columns))


class PackedLinear(nn.Module):
    def __init__(self, packed, reduction, layout, partition):
        super().__init__()
        if packed.dtype != torch.int8 or packed.ndim != 1 or packed.stride() != (1,):
            raise ValueError("Gemma packed weight views require unit-stride INT8 bytes")
        if partition["columns"] <= 0:
            raise ValueError("Empty Gemma panel ranges have no compute task")
        _panel_linear_fake(
            torch.empty((1, reduction)),
            packed.narrow(0, partition["byte_start"], partition["bytes"]),
            partition["columns"],
            layout["tile_m"],
            layout["tile_n"],
            layout["tile_k"],
            layout["bank_bytes"],
        )
        self.register_buffer(
            "weight", packed.narrow(0, partition["byte_start"], partition["bytes"])
        )
        self.reduction, self.columns = reduction, partition["columns"]
        self.layout = dict(layout)

    def forward(self, values):
        result = panel_linear(
            values.reshape(-1, self.reduction),
            self.weight,
            self.columns,
            self.layout["tile_m"],
            self.layout["tile_n"],
            self.layout["tile_k"],
            self.layout["bank_bytes"],
        )
        return result.reshape(*values.shape[:-1], self.columns)


def register(compiler):
    from buddy.compiler.graph.operation import OpType
    from stack.compiler.quant.mxfp8_graph import MXFP8MatmulOp, lower_matmul

    class PanelLinearOp(MXFP8MatmulOp):
        def __init__(self):
            super().__init__()
            self._op_type = OpType.ReduceType

    def lower(node, symbols):
        packed = MXFP8MatmulOp()
        packed._arguments = node.args[:2]
        packed._tensor_meta = dict(node.tensor_meta)
        columns, tile_m, tile_n, tile_k, bank_bytes = node.args[2:]
        if columns != node.tensor_meta["shape"][1]:
            raise ValueError("Gemma panel task output columns mismatch")
        packed.layout = dict(
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, bank_bytes=bank_bytes
        )
        return lower_matmul(packed, symbols)

    compiler._ops_map["panel_linear.default"] = PanelLinearOp
    compiler._ops_registry["PanelLinearOp"] = lower
