"""Integer GELU on MXINT8 blocks; floating point is only the offline boundary."""

import torch
from torch.overrides import TorchFunctionMode

from compiler.frontend.quant import dequantize_mxint8, quantize_mxint8


def round_shift(value, shift):
    shift = torch.as_tensor(shift, device=value.device).clamp(0, 62)
    magnitude = value.abs()
    quotient = magnitude >> shift
    remainder = magnitude & ((1 << shift) - 1)
    half = 1 << (shift - 1).clamp_min(0)
    increment = (shift > 0) & (
        (remainder > half) | ((remainder == half) & ((quotient & 1) != 0))
    )
    rounded = quotient + increment.to(torch.int64)
    return torch.where(value < 0, -rounded, rounded)


def activate(packed, table, gate):
    """One 16 KiB Q26 table covers [-16,16), with 1/128 spacing."""
    k = packed.shape[-1] * 32 // 33
    codes = packed[..., :k].to(torch.int64).reshape(*packed.shape[:-1], k // 32, 32)
    assert torch.all(codes >= -127), "MXINT8 activation codes must lie in [-127,127]"
    exponent = (
        packed[..., k:].contiguous().view(torch.uint8).to(torch.int64)[..., None] - 127
    )
    assert torch.all(exponent < 128), "MXINT8 activation requires finite exponents"
    # Large magnitudes use the integer tails; tiny blocks use GELU's linear term.
    x = codes << (exponent.clamp(-24, 4) + 26)
    position = x + (16 << 26)
    index = (position >> 19).clamp(0, 4095)
    first = table[index].to(torch.int64)
    last = torch.where(
        index == 4095, 16 << 26, table[(index + 1).clamp_max(4095)].to(torch.int64)
    )
    fraction = position & ((1 << 19) - 1)
    value = first + round_shift((last - first) * fraction, 19)
    value = torch.where(
        exponent >= -10,
        round_shift(value, exponent + 10),
        value << (-exponent - 10).clamp(0, 14),
    )
    value = torch.where(
        x >= (16 << 26), codes << 16, torch.where(x <= -(16 << 26), 0, value)
    )
    value = torch.where(exponent < -24, codes << 15, value)
    if gate is not None:
        assert (
            gate.shape == packed.shape
        ), "GELU and gate blocks must have the same shape"
        gate_codes = gate[..., :k].to(torch.int64).reshape(codes.shape)
        gate_exponent = (
            gate[..., k:].contiguous().view(torch.uint8).to(torch.int64)[..., None]
            - 127
        )
        assert torch.all(gate_codes >= -127) and torch.all(gate_exponent < 128)
        value *= gate_codes
        exponent = exponent + gate_exponent
    maximum = value.abs().amax(-1, keepdim=True)
    remaining, highest = maximum, torch.zeros_like(maximum)
    for shift in [16, 8, 4, 2, 1]:
        upper = remaining >= (1 << shift)
        highest += upper.to(torch.int64) * shift
        remaining = torch.where(upper, remaining >> shift, remaining)
    output_exponent = (exponent + highest - 22).clamp(-127, 127)
    output_exponent = torch.where(maximum == 0, 0, output_exponent)
    shift = output_exponent - exponent + 16
    result = torch.where(
        shift < -6,
        value.sign() * 127,
        torch.where(
            shift >= 0, round_shift(value, shift), value << (-shift).clamp(0, 6)
        ),
    ).clamp(-127, 127)
    result = result.reshape(*packed.shape[:-1], k).to(torch.int8)
    scales = (output_exponent[..., 0] + 127).to(torch.uint8).view(torch.int8)
    return torch.cat([result, scales], -1)


class IntegerActivations(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        x = torch.arange(4096, dtype=torch.float64) / 128 - 16
        # Table generation is offline; activate() uses only integer operations.
        self.table = torch.round(torch.nn.functional.gelu(x) * (1 << 26)).to(
            torch.int32
        )
        self.calls = 0
        self.gated_calls = 0
        self.pending = None

    def __torch_function__(self, function, types, args=(), kwargs=None):
        if getattr(function, "__name__", None) == "gelu":
            assert not kwargs or kwargs.get("approximate", "none") == "none"
            self.calls += 1
            packed = quantize_mxint8(args[0], -1)
            output = activate(packed, self.table, None)
            result = dequantize_mxint8(output, -1)[..., : args[0].shape[-1]]
            self.pending = (result, packed)
            return result
        if (
            getattr(function, "__name__", None) in {"mul", "__mul__"}
            and self.pending is not None
        ):
            matches = [
                index for index, value in enumerate(args) if value is self.pending[0]
            ]
            if matches:
                assert len(args) == 2 and isinstance(args[1 - matches[0]], torch.Tensor)
                gate = quantize_mxint8(args[1 - matches[0]], -1)
                output = activate(self.pending[1], self.table, gate)
                self.pending = None
                self.gated_calls += 1
                return dequantize_mxint8(output, -1)[
                    ..., : args[1 - matches[0]].shape[-1]
                ]
        return function(*args, **(kwargs or {}))


def integer_activations(graph, params):
    """The recipe chooses quantization before placing the model on harts."""
    from buddy.compiler.graph import NodeType
    from buddy.compiler.graph.operation import (
        AddOp,
        CloneOp,
        ErfOp,
        MulOp,
        PermuteOp,
        PlaceholderOp,
        ViewOp,
    )
    from buddy.compiler.graph.transform.quantization.quantize import sort_graph
    from buddy.compiler.graph.type import TensorDType
    from compiler.frontend.quant import (
        FP32ToMXINT8QuantOp,
        MXINT8ActivationOp,
        MXINT8ToFP32DequantOp,
    )
    from frontend.norm import discard, replace
    from frontend.quant.graph import operation, packed

    values = dict(zip((node.name for node in graph.params), params, strict=True))
    table = PlaceholderOp()
    table.name = "integer_gelu_table"
    table.tensor_meta = {"shape": [4096], "dtype": TensorDType.Int32}
    graph.add_node(table, NodeType.FakeNode)
    values[table.name] = IntegerActivations().table
    count = gated_count = 0
    for erf in [node for node in graph.body if isinstance(node, ErfOp)]:
        scaled = graph.node_table[erf.args[0]]
        (name,) = erf.children
        added = graph.node_table[name]
        (name,) = added.children
        gelu = graph.node_table[name]
        if not (
            isinstance(scaled, MulOp)
            and scaled.args[1] == 0.7071067811865476
            and isinstance(added, AddOp)
            and added.args == [erf.name, 1]
            and isinstance(gelu, MulOp)
        ):
            raise ValueError(f"{erf.name}: expected the GELU expression")
        source = graph.node_table[scaled.args[0]]
        half_name = gelu.args[1] if gelu.args[0] == added.name else gelu.args[0]
        half = graph.node_table[half_name]
        if not isinstance(half, MulOp) or half.args != [source.name, 0.5]:
            raise ValueError(f"{gelu.name}: expected GELU's half input")
        target, gate, gated = gelu, source, False
        (name,) = gelu.children
        child = graph.node_table[name]
        if isinstance(child, MulOp) and all(isinstance(arg, str) for arg in child.args):
            gate = graph.node_table[
                child.args[1] if child.args[0] == gelu.name else child.args[0]
            ]
            if gate.tensor_meta["shape"] != source.tensor_meta["shape"]:
                raise ValueError(f"{child.name}: gated GELU requires matching rows")
            target, gated = child, True
        shape = list(source.tensor_meta["shape"])
        axis = len(shape) - 1
        if shape[-1] % 32:
            raise ValueError(
                f"{target.name}: integer GELU channels must form complete MXINT8 blocks"
            )
        codes = operation(
            graph,
            FP32ToMXINT8QuantOp,
            f"{target.name}_input_mxint8",
            [source],
            [axis],
            packed(shape, axis),
            TensorDType.Int8,
        )
        gate_codes = (
            operation(
                graph,
                FP32ToMXINT8QuantOp,
                f"{target.name}_gate_mxint8",
                [gate],
                [axis],
                packed(shape, axis),
                TensorDType.Int8,
            )
            if gated
            else codes
        )
        output = operation(
            graph,
            MXINT8ActivationOp,
            f"{target.name}_integer_mxint8",
            [codes, gate_codes, table],
            [axis, gated],
            packed(shape, axis),
            TensorDType.Int8,
        )
        parents = replace(graph, target, MXINT8ToFP32DequantOp(), [output], [axis])
        discard(graph, parents)
        count += 1
        gated_count += gated
    if not count:
        raise ValueError("Laya graph contains no GELU expressions")
    # Matrix consumers use the calibrated activation's packed output directly.
    # Move pure row reshapes before quantization so sharding keeps MXINT8 blocks.
    for quant in list(graph.body):
        if (
            not isinstance(quant, FP32ToMXINT8QuantOp)
            or graph.node_table.get(quant.name) is not quant
        ):
            continue
        source = graph.node_table[quant.args[0]]
        while isinstance(source, (CloneOp, ViewOp, PermuteOp)):
            parent = graph.node_table[source.args[0]]
            if isinstance(source, PermuteOp):
                shape, axes = parent.tensor_meta["shape"], source.args[1]
                if axes[-1] != len(shape) - 1 or [a for a in axes if shape[a] != 1] != [
                    a for a, size in enumerate(shape) if size != 1
                ]:
                    break
            if source.tensor_meta["shape"][-1] != parent.tensor_meta["shape"][-1]:
                break
            source = parent
        if not isinstance(source, MXINT8ToFP32DequantOp):
            continue
        activation = graph.node_table[source.args[0]]
        if not isinstance(activation, MXINT8ActivationOp):
            continue
        shape = list(graph.node_table[quant.args[0]].tensor_meta["shape"])
        axis = len(shape) - 1
        if (
            quant.args[1] != axis
            or source.args[1] != len(source.tensor_meta["shape"]) - 1
        ):
            raise ValueError(
                f"{quant.name}: integer activation requires last-axis MXINT8"
            )
        inputs = []
        for index in range(2 if activation.args[4] else 1):
            incoming = graph.node_table[activation.args[index]]
            if not isinstance(incoming, FP32ToMXINT8QuantOp):
                raise ValueError(
                    f"{activation.name}: expected quantized activation inputs"
                )
            value = graph.node_table[incoming.args[0]]
            if list(value.tensor_meta["shape"]) != shape:
                sizes = [
                    size.node.expr if isinstance(size, torch.SymInt) else size
                    for size in shape
                ]
                value = operation(
                    graph,
                    ViewOp,
                    f"{quant.name}_activation_row{index}",
                    [value],
                    [sizes],
                    shape,
                    value.tensor_meta["dtype"],
                )
            inputs.append(
                operation(
                    graph,
                    FP32ToMXINT8QuantOp,
                    f"{quant.name}_activation_mx{index}",
                    [value],
                    [axis],
                    packed(shape, axis),
                    TensorDType.Int8,
                )
            )
        if len(inputs) == 1:
            inputs.append(inputs[0])
        table = graph.node_table[activation.args[2]]
        parents = replace(
            graph,
            quant,
            MXINT8ActivationOp(),
            [*inputs, table],
            [axis, activation.args[4]],
        )
        discard(graph, parents)
    sort_graph(graph)
    print(
        f"Mapped {count} GELU expressions to integer MXINT8, including {gated_count} gates"
    )
    return [values[node.name] for node in graph.params]
