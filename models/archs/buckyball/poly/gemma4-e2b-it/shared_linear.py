import math

import torch
from torch import nn


@torch.library.custom_op("gemma::shared_linear", mutates_args=())
def shared_linear(
    values: torch.Tensor,
    packed: torch.Tensor,
    destination: torch.Tensor,
    logical_columns: int,
    tile_n: int,
    tile_k: int,
    bank_bytes: int,
    shared_bytes: int,
    mode: str,
) -> torch.Tensor:
    # Capture returns only the control token; bank data is produced by the NPU.
    return destination.clone()


@shared_linear.register_fake
def _shared_linear_fake(
    values,
    packed,
    destination,
    logical_columns,
    tile_n,
    tile_k,
    bank_bytes,
    shared_bytes,
    mode,
):
    if (
        values.ndim != 2
        or values.dtype != torch.float32
        or values.shape[0] not in (1, 16)
        or values.shape[1] <= 0
        or values.shape[1] % 32
        or packed.ndim != 1
        or packed.dtype != torch.int8
        or packed.stride() != (1,)
        or destination.shape != (1,)
        or destination.dtype != torch.int64
        or tile_n <= 0
        or tile_n % 16
        or tile_k <= 0
        or tile_k % 32
        or tile_k >= 4096
        or not 0 < logical_columns <= tile_n
        or mode not in ("panel", "window")
        or packed.numel() != math.ceil(values.shape[1] / tile_k) * bank_bytes
        or tile_n * min(tile_k, values.shape[1]) * 33 // 32 > bank_bytes
        or values.shape[0] * tile_n * 4 > shared_bytes
        or mode == "panel"
        and values.shape[0] * min(tile_k, values.shape[1]) * 33 // 32 > bank_bytes
        or mode == "window"
        and (values.numel() * 33 // 32 > bank_bytes or values.shape[1] > 65535)
    ):
        raise ValueError(
            "Gemma shared linear requires M=1 or M=16 and a valid explicit panel plan"
        )
    return destination.new_empty((1,))


class SharedLinear(nn.Module):
    def __init__(
        self, packed, reduction, layout, logical_columns, *, mode, shared_bytes
    ):
        super().__init__()
        self.register_buffer("weight", packed)
        self.reduction = reduction
        self.logical_columns = logical_columns
        self.tile_n = layout["tile_n"]
        self.tile_k = layout["tile_k"]
        self.bank_bytes = layout["bank_bytes"]
        self.shared_bytes = shared_bytes
        self.mode = mode
        _shared_linear_fake(
            torch.empty((1, reduction)),
            packed,
            torch.empty((1,), dtype=torch.int64),
            logical_columns,
            self.tile_n,
            self.tile_k,
            self.bank_bytes,
            shared_bytes,
            mode,
        )

    def forward(self, values, destination):
        if values.shape[1] != self.reduction:
            raise ValueError("Gemma shared linear reduction differs from its weight")
        return shared_linear(
            values,
            self.weight,
            destination,
            self.logical_columns,
            self.tile_n,
            self.tile_k,
            self.bank_bytes,
            self.shared_bytes,
            self.mode,
        )


def register(compiler):
    from buddy.compiler.graph.operation import Op, OpType
    from buddy_mlir import ir
    from buddy_mlir.dialects import arith, bufferization, tensor

    class SharedLinearOp(Op):
        def __init__(self):
            super().__init__()
            self._op_type = OpType.ReduceType

    def lower(node, symbols):
        values, weight, destination = [symbols[(name, 0)] for name in node.args[:3]]
        columns, tile_n, tile_k, bank_bytes, shared_bytes, mode = node.args[3:]
        shape = list(ir.RankedTensorType(values.type).shape)
        if len(shape) != 2 or any(d <= 0 for d in shape) or shape[0] not in (1, 16):
            raise ValueError(
                "Gemma shared linear requires a static matrix with M=1 or M=16"
            )
        rows, reduction = shape
        if reduction % 32 or mode not in ("panel", "window"):
            raise ValueError("Gemma shared linear requires an explicit MXFP8 mode")
        if mode == "window" and (
            rows * reduction * 33 // 32 > bank_bytes or reduction > 65535
        ):
            raise ValueError(
                "Gemma shared A-window exceeds its private bank or 16-bit full-K limit"
            )
        if not 0 < columns <= tile_n:
            raise ValueError("Gemma logical columns exceed its original physical panel")
        integer = ir.IntegerType.get_signless(64)
        chunk = reduction if mode == "window" else tile_k
        size = math.ceil(reduction / chunk) * bank_bytes
        quantized = ir.Operation.create(
            "buckyball.mxfp8_quant",
            operands=[values],
            results=[ir.RankedTensorType.get([size], ir.IntegerType.get_signless(8))],
            attributes={
                "tile_rows": ir.IntegerAttr.get(integer, rows),
                "tile_k": ir.IntegerAttr.get(integer, chunk),
                "bank_bytes": ir.IntegerAttr.get(integer, bank_bytes),
            },
        ).result
        buffers = []
        for value in (quantized, weight):
            ranked = ir.RankedTensorType(value.type)
            layout = ir.StridedLayoutAttr.get(ir.ShapedType.get_dynamic_size(), [1])
            buffers.append(
                bufferization.ToBufferOp(
                    ir.MemRefType.get(ranked.shape, ranked.element_type, layout),
                    value,
                    read_only=True,
                ).result
            )
        zero = arith.ConstantOp(ir.IndexType.get(), 0).result
        bank = tensor.ExtractOp(destination, [zero]).result
        result = ir.Operation.create(
            "buckyball.mxfp8_shared_" + mode,
            operands=[*buffers, bank],
            results=[integer],
            attributes={
                name: ir.IntegerAttr.get(integer, value)
                for name, value in {
                    "rows": rows,
                    "columns": tile_n,
                    "reduction_k": reduction,
                    "tile_k": tile_k,
                    "bank_bytes": bank_bytes,
                    "shared_bytes": shared_bytes,
                }.items()
            },
        ).result
        return tensor.FromElementsOp(
            ir.RankedTensorType.get([1], integer), [result]
        ).result

    compiler._ops_map["shared_linear.default"] = SharedLinearOp
    compiler._ops_registry["SharedLinearOp"] = lower
