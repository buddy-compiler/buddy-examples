import math
import torch
from buddy.compiler.graph.operation import (
    AddOp,
    ClampOp,
    CloneOp,
    Conv2dOp,
    DivOp,
    MatmulOp,
    MeanOp,
    MulOp,
    OutputOp,
    PermuteOp,
    ReluOp,
    SliceOp,
    ViewOp,
)
from buddy.compiler.graph.transform.quantization.quantize import sort_graph
from buddy.compiler.graph.type import TensorDType
from compiler.frontend.quant import (
    FP32ToInt8QuantOp,
    Int32ToInt8QuantOp,
    Int32ToFP32DequantOp,
    FixedRequantOp,
    FixedUnaryOp,
    FixedBinaryOp,
    FixedMeanOp,
)
from frontend.quant.graph import operation, parameter, replace_argument

FRACTION = 24


def coefficients(scale):
    mantissa, exponent = torch.frexp(scale.double().reshape(-1) * (1 << FRACTION))
    multiplier = torch.round(mantissa * (1 << 30)).to(torch.int64)
    shift = 30 - exponent.to(torch.int64)
    if not torch.all(torch.isfinite(scale) & (scale >= 0)) or torch.any(shift < 0):
        raise ValueError("Fixed-point coefficient is outside the ISA range")
    # At this shift every signed 32-bit input rounds to zero, including ties.
    multiplier = torch.where(shift >= 62, 0, multiplier)
    shift = torch.where(shift >= 62, 0, shift)
    return multiplier.to(torch.int32), shift.to(torch.int32)


def fixed(graph, params):
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    codes, steps = {}, {}
    matmuls = [node for node in graph.body if isinstance(node, MatmulOp)]
    final_matmul = matmuls[-1]

    def publish(old, code, users):
        rank = len(old.tensor_meta["shape"])
        if rank not in steps:
            steps[rank] = parameter(
                graph,
                values,
                f"fixed_step_{rank}",
                torch.full([1] * rank, 2.0**-FRACTION),
            )
        decoded = operation(
            graph,
            Int32ToFP32DequantOp,
            old.name + "_fixed_values",
            [code, steps[rank]],
            old.tensor_meta["shape"],
            TensorDType.Float32,
        )
        for name in users:
            user = graph.node_table[name]
            user._arguments = [
                decoded.name if arg == old.name else arg for arg in user.args
            ]
            user._parents = [
                decoded.name if parent == old.name else parent
                for parent in user.parents
            ]
            old.children.remove(name)
            decoded.add_children(name)
        codes[decoded.name] = code

    for node in list(graph.body):
        if isinstance(node, FP32ToInt8QuantOp):
            if node.args[0] in codes:
                multiplier = parameter(
                    graph,
                    values,
                    node.name + "_fixed_multiplier",
                    values[node.args[1]] * (2.0**-FRACTION),
                )
                replace_argument(graph, node, 0, codes[node.args[0]])
                replace_argument(graph, node, 1, multiplier)
                replacement = Int32ToInt8QuantOp()
                replacement.name = node.name
                graph.displace_node(node, replacement)
            continue
        if node.tensor_meta.get("dtype") != TensorDType.Float32:
            continue
        if isinstance(node, (Conv2dOp, MatmulOp)):
            convolution = isinstance(node, Conv2dOp)
            bias_user = node if convolution else graph.node_table[node.children[0]]
            bias_index = 2 if convolution else 0
            bias_value = graph.node_table[bias_user.args[bias_index]]
            assert isinstance(bias_value, Int32ToFP32DequantOp)
            scale = graph.node_table[bias_value.args[1]]
            if isinstance(scale, ViewOp):
                scale = graph.node_table[scale.args[0]]
            assert isinstance(scale, MulOp)
            scale_value = (values[scale.args[0]] * values[scale.args[1]]).reshape(-1)
            for index in (0, 1):
                dequant = graph.node_table[node.args[index]]
                replace_argument(graph, node, index, graph.node_table[dequant.args[0]])
            replace_argument(
                graph, bias_user, bias_index, graph.node_table[bias_value.args[0]]
            )
            node.tensor_meta["dtype"] = TensorDType.Int32
            result = node if convolution else bias_user
            users = list(result.children)
            result.tensor_meta["dtype"] = TensorDType.Int32
            if node is final_matmul:
                step = parameter(
                    graph,
                    values,
                    node.name + "_output_step",
                    scale_value.reshape(1, -1),
                )
                decoded = operation(
                    graph,
                    Int32ToFP32DequantOp,
                    result.name + "_output",
                    [result, step],
                    result.tensor_meta["shape"],
                    TensorDType.Float32,
                )
                for name in users:
                    graph.replace_as_parent(result, [name], decoded)
                    result.children.remove(name)
                    decoded.add_children(name)
                continue
            multiplier, shift = coefficients(scale_value)
            padded = -multiplier.numel() % 16
            m = parameter(
                graph,
                values,
                node.name + "_fixed_m",
                torch.nn.functional.pad(multiplier, (0, padded)),
            )
            s = parameter(
                graph,
                values,
                node.name + "_fixed_s",
                torch.nn.functional.pad(shift, (0, padded)),
            )
            code = operation(
                graph,
                FixedRequantOp,
                result.name + "_fixed",
                [result, m, s],
                result.tensor_meta["shape"],
                TensorDType.Int32,
            )
            code.add_argument(FRACTION)
            publish(result, code, users)
            continue

        source = graph.node_table[node.args[0]] if node.parents else None
        code = None
        if isinstance(node, DivOp) and node.args[1] == 6:
            product = source
            kind = "hardswish" if isinstance(product, MulOp) else "hardsigmoid"
            high = graph.node_table[product.args[1]] if kind == "hardswish" else product
            if not isinstance(high, ClampOp) or high.args[1:] != [None, 6]:
                continue
            low = graph.node_table[high.args[0]]
            if not isinstance(low, ClampOp) or low.args[1:] != [0]:
                continue
            add = graph.node_table[low.args[0]]
            if (
                not isinstance(add, AddOp)
                or add.args[1] != 3
                or add.args[0] not in codes
            ):
                continue
            if kind == "hardswish" and product.args[0] != add.args[0]:
                continue
            code = operation(
                graph,
                FixedUnaryOp,
                node.name + "_fixed",
                [codes[add.args[0]]],
                node.tensor_meta["shape"],
                TensorDType.Int32,
            )
            code.add_argument(kind)
            code.add_argument(FRACTION)
        elif isinstance(node, ReluOp) and node.args[0] in codes:
            code = operation(
                graph,
                FixedUnaryOp,
                node.name + "_fixed",
                [codes[node.args[0]]],
                node.tensor_meta["shape"],
                TensorDType.Int32,
            )
            code.add_argument("relu")
            code.add_argument(FRACTION)
        elif isinstance(node, (AddOp, MulOp)) and all(
            isinstance(arg, str) and arg in codes for arg in node.args[:2]
        ):
            operands = [codes[arg] for arg in node.args[:2]]
            if operands[0].tensor_meta["shape"] != node.tensor_meta["shape"]:
                operands.reverse()
            code = operation(
                graph,
                FixedBinaryOp,
                node.name + "_fixed",
                operands,
                node.tensor_meta["shape"],
                TensorDType.Int32,
            )
            code.add_argument("add" if isinstance(node, AddOp) else "mul")
            code.add_argument(FRACTION)
        elif isinstance(node, MeanOp) and node.args[0] in codes:
            if node.args[1:] != [[1, 2], True]:
                raise ValueError(
                    "Fixed mean requires NHWC spatial reduction with kept dimensions"
                )
            code = operation(
                graph,
                FixedMeanOp,
                node.name + "_fixed",
                [codes[node.args[0]]],
                node.tensor_meta["shape"],
                TensorDType.Int32,
            )
        elif (
            isinstance(node, (ViewOp, SliceOp, PermuteOp, CloneOp))
            and node.args[0] in codes
        ):
            code = operation(
                graph,
                type(node),
                node.name + "_fixed",
                [codes[node.args[0]]],
                node.tensor_meta["shape"],
                TensorDType.Int32,
            )
            for argument in node.args[1:]:
                code.add_argument(argument)
        if code is not None:
            publish(node, code, list(node.children))

    # The quantizer reads a logical channel vector; DMA reads whole 16-channel
    # blocks from its padded parameter, without a host-side packing buffer.
    for node in list(graph.body):
        if not isinstance(node, (FP32ToInt8QuantOp, Int32ToInt8QuantOp)):
            continue
        multiplier = graph.node_table[node.args[1]]
        shape = multiplier.tensor_meta["shape"]
        channels = shape[-1]
        input_channels = graph.node_table[node.args[0]].tensor_meta["shape"][-1]
        value = values[multiplier.name]
        if torch.all(value == value.reshape(-1)[0]):
            # Keep scalar broadcasting visible; store full bank-readable rows.
            shape = [1] * len(shape)
            channels = 1
            physical_shape = [*shape[:-1], (input_channels + 15) // 16 * 16]
            value = value.reshape(-1)[0].expand(physical_shape).contiguous()
        else:
            if channels % 16 == 0:
                continue
            repeats = [1] * value.ndim
            repeats[-1] = math.lcm(channels, 16) // channels
            value = value.repeat(repeats)
        padded = parameter(
            graph,
            values,
            multiplier.name + "_padded",
            value,
        )
        logical = operation(
            graph,
            SliceOp,
            multiplier.name + "_channels",
            [padded],
            shape,
            TensorDType.Float32,
        )
        for argument in (len(shape) - 1, 0, channels):
            logical.add_argument(argument)
        replace_argument(graph, node, 1, logical)

    while True:
        dead = [
            node
            for node in graph.body
            if not node.children and not isinstance(node, OutputOp)
        ]
        if not dead:
            break
        for node in reversed(dead):
            graph.delete_node(
                node, [graph.node_table[parent] for parent in node.parents]
            )
    sort_graph(graph)
    return [values[node.name] for node in graph.params]
