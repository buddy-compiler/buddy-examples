"""Qwen3-0.6B calibrated integer SiLU policy and graph mapping."""

import torch
from torch.overrides import TorchFunctionMode

from compiler.frontend.quant import dequantize_mxint8, quantize_mxint8
from buddy.compiler.graph.operation import OutputOp, PlaceholderOp


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
    # Large magnitudes use the integer tails; tiny blocks use SiLU's linear term.
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
        ), "SiLU and gate blocks must have the same shape"
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


def replace(graph, old, new, arguments, attributes):
    parents = [graph.node_table[name] for name in old.parents]
    category = new._op_type
    new.name = old.name
    graph.displace_node(old, new)
    new._op_type = category
    for parent in parents:
        parent.children.remove(new.name)
    new.args.clear()
    new.parents.clear()
    new.kwargs.clear()
    for argument in arguments:
        new.add_argument(argument.name)
        new.add_parent(argument.name)
        argument.add_children(new.name)
    for attribute in attributes:
        new.add_argument(attribute)
    return parents


def discard(graph, pending):
    while pending:
        node = pending.pop()
        if (
            node.name not in graph.node_table
            or node.children
            or isinstance(node, (PlaceholderOp, OutputOp))
        ):
            continue
        parents = [graph.node_table[name] for name in node.parents]
        graph.delete_node(node, parents)
        pending.extend(parents)


class IntegerActivations(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        x = torch.arange(4096, dtype=torch.float64) / 128 - 16
        self.table = torch.round(torch.nn.functional.silu(x) * (1 << 26)).to(
            torch.int32
        )
        self.calls = 0
        self.pending = None

    def __torch_function__(self, function, types, args=(), kwargs=None):
        name = getattr(function, "__name__", None)
        if name == "silu":
            layer = self.calls % 28
            self.calls += 1
            # One retained layer per seven-layer group meets the 1% accuracy gate.
            if layer % 7 == 0:
                self.pending = None
                return function(*args, **(kwargs or {}))
            packed = quantize_mxint8(args[0], -1)
            output = activate(packed, self.table, None)
            result = dequantize_mxint8(output, -1)[..., : args[0].shape[-1]]
            self.pending = (result, packed)
            return result
        if name in {"mul", "__mul__"} and self.pending is not None:
            matches = [i for i, value in enumerate(args) if value is self.pending[0]]
            if matches:
                assert len(args) == 2
                gate = quantize_mxint8(args[1 - matches[0]], -1)
                output = activate(self.pending[1], self.table, gate)
                self.pending = None
                return dequantize_mxint8(output, -1)[
                    ..., : args[1 - matches[0]].shape[-1]
                ]
        return function(*args, **(kwargs or {}))


def integer_activations(graph, params):
    from buddy.compiler.graph import NodeType
    from buddy.compiler.graph.operation import (
        AddOp,
        CloneOp,
        DivOp,
        ExpOp,
        MulOp,
        NegOp,
        PermuteOp,
        ViewOp,
    )
    from buddy.compiler.graph.transform.quantization.quantize import sort_graph
    from buddy.compiler.graph.type import TensorDType
    from compiler.frontend.quant import (
        FP32ToMXINT8QuantOp,
        MXINT8ActivationOp,
        MXINT8ToFP32DequantOp,
    )
    from frontend.quant.graph import operation, packed

    values = dict(zip((node.name for node in graph.params), params, strict=True))
    table = PlaceholderOp()
    table.name = "integer_silu_table"
    table.tensor_meta = {"shape": [4096], "dtype": TensorDType.Int32}
    graph.add_node(table, NodeType.FakeNode)
    values[table.name] = IntegerActivations().table
    expressions = []
    for division in list(graph.body):
        if not isinstance(division, DivOp) or not isinstance(division.args[1], str):
            continue
        denominator = graph.node_table[division.args[1]]
        if not isinstance(denominator, AddOp) or denominator.args[1] != 1:
            continue
        exponential = graph.node_table[denominator.args[0]]
        if not isinstance(exponential, ExpOp):
            continue
        negative = graph.node_table[exponential.args[0]]
        if not isinstance(negative, NegOp) or negative.args[0] != division.args[0]:
            raise ValueError(
                f"{division.name}: expected SiLU's input divided by 1 + exp(-input)"
            )
        expressions.append(division)
    if len(expressions) != 28:
        raise ValueError(
            f"Expected 28 Qwen3-0.6B SiLU expressions, got {len(expressions)}"
        )
    for layer, silu in enumerate(expressions):
        if layer % 7 == 0:
            continue
        source = graph.node_table[silu.args[0]]
        (name,) = silu.children
        target = graph.node_table[name]
        if not isinstance(target, MulOp) or not all(
            isinstance(arg, str) for arg in target.args
        ):
            raise ValueError(f"{silu.name}: expected a gated SiLU")
        gate = graph.node_table[
            target.args[1] if target.args[0] == silu.name else target.args[0]
        ]
        shape = list(source.tensor_meta["shape"])
        if gate.tensor_meta["shape"] != source.tensor_meta["shape"] or shape[-1] % 32:
            raise ValueError(
                f"{target.name}: gated SiLU requires matching complete MXINT8 rows"
            )
        axis = len(shape) - 1
        codes = operation(
            graph,
            FP32ToMXINT8QuantOp,
            f"{target.name}_input_mxint8",
            [source],
            [axis],
            packed(shape, axis),
            TensorDType.Int8,
        )
        gate_codes = operation(
            graph,
            FP32ToMXINT8QuantOp,
            f"{target.name}_gate_mxint8",
            [gate],
            [axis],
            packed(shape, axis),
            TensorDType.Int8,
        )
        output = operation(
            graph,
            MXINT8ActivationOp,
            f"{target.name}_integer_mxint8",
            [codes, gate_codes, table],
            [axis, True],
            packed(shape, axis),
            TensorDType.Int8,
        )
        parents = replace(graph, target, MXINT8ToFP32DequantOp(), [output], [axis])
        discard(graph, parents)
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
        "Mapped 24 gated SiLU expressions to integer MXINT8; layers 0, 7, 14, 21 retain FP32"
    )
    return [values[node.name] for node in graph.params]
