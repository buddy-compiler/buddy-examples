import torch
import buddy_mlir.ir as ir
from buddy.compiler.graph import NodeType
from buddy.compiler.graph.operation import (
    AddOp,
    CloneOp,
    GetItemOp,
    MulOp,
    Op,
    OpType,
    OutputOp,
    PermuteOp,
    PlaceholderOp,
    RsqrtOp,
    SubOp,
    VarMeanOp,
    ViewOp,
)
from buddy.compiler.graph.sizes import mlir_shape
from buddy.compiler.graph.type import TensorDType
from buddy.compiler.graph.transform.quantization.quantize import sort_graph
from compiler.frontend.quant import FP32ToMXINT8QuantOp, MXINT8ToFP32DequantOp
from frontend.quant.graph import operation, packed


class FixedNormInt8Op(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ReduceType


def fixed_norm_op(node, symbol_table):
    return ir.Operation.create(
        "frontend.fixed_norm_mxint8",
        results=[
            ir.RankedTensorType.get(
                mlir_shape(node.tensor_meta["shape"]), ir.IntegerType.get_signless(8)
            )
        ],
        operands=[symbol_table[(name, 0)] for name in node.args],
    ).result


ops_registry = {"FixedNormInt8Op": fixed_norm_op}


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


def integer_norm(graph, params):
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    count = 0
    for variance_mean in list(graph.body):
        if not isinstance(variance_mean, VarMeanOp):
            continue
        source = graph.node_table[variance_mean.args[0]]
        shape = list(source.tensor_meta["shape"])
        k = shape[-1]
        if (
            k % 512
            or variance_mean.args[1] != [len(shape) - 1]
            or variance_mean.kwargs != {"correction": 0, "keepdim": True}
        ):
            raise ValueError(
                f"{variance_mean.name}: expected last-axis LayerNorm in complete 512-element groups"
            )
        parts = {}
        for name in variance_mean.children:
            node = graph.node_table[name]
            if not isinstance(node, GetItemOp):
                raise ValueError(
                    f"{name}: expected a LayerNorm variance/mean projection"
                )
            parts[node.args[1]] = node
        variance, mean = parts[0], parts[1]
        (name,) = variance.children
        epsilon = graph.node_table[name]
        (name,) = epsilon.children
        inverse = graph.node_table[name]
        (name,) = mean.children
        centered = graph.node_table[name]
        (name,) = inverse.children
        scaled = graph.node_table[name]
        (name,) = scaled.children
        affine = graph.node_table[name]
        if not (
            isinstance(epsilon, AddOp)
            and epsilon.args[0] == variance.name
            and isinstance(inverse, RsqrtOp)
            and isinstance(centered, SubOp)
            and centered.args == [source.name, mean.name]
            and isinstance(scaled, MulOp)
            and set(scaled.args) == {centered.name, inverse.name}
            and isinstance(affine, MulOp)
        ):
            raise ValueError(f"{variance_mean.name}: unrecognized LayerNorm expression")
        gamma_name = affine.args[1] if affine.args[0] == scaled.name else affine.args[0]
        gamma = graph.node_table[gamma_name]
        if not isinstance(gamma, PlaceholderOp) or list(gamma.tensor_meta["shape"]) != [
            k
        ]:
            raise ValueError(f"{affine.name}: expected a static LayerNorm scale")
        target = affine
        beta = torch.zeros(k, dtype=torch.int32)
        (name,) = affine.children
        child = graph.node_table[name]
        if isinstance(child, AddOp):
            other_name = (
                child.args[1] if child.args[0] == affine.name else child.args[0]
            )
            other = graph.node_table[other_name]
            if isinstance(other, PlaceholderOp) and list(
                other.tensor_meta["shape"]
            ) == [k]:
                beta = torch.round(values[other.name].detach() * 65536).to(torch.int32)
                target = child
        eps = round(epsilon.args[1] * (1 << 32))
        if not 0 < eps < (1 << 64):
            raise ValueError(f"{epsilon.name}: epsilon does not fit positive Q32")
        header = torch.tensor(
            [eps & 0xFFFFFFFF, eps >> 32, 0, 0], dtype=torch.int64
        ).to(torch.int32)
        payload = torch.cat(
            [
                header,
                torch.round(values[gamma_name].detach() * 65536).to(torch.int32),
                beta,
            ]
        ).view(torch.int8)
        coefficients = PlaceholderOp()
        coefficients.name = f"{target.name}_norm_coefficients"
        coefficients.tensor_meta = {
            "shape": [payload.numel()],
            "dtype": TensorDType.Int8,
        }
        graph.add_node(coefficients, NodeType.FakeNode)
        values[coefficients.name] = payload
        codes = operation(
            graph,
            FixedNormInt8Op,
            f"{target.name}_fixed_codes",
            [source, coefficients],
            [],
            packed(shape, len(shape) - 1),
            TensorDType.Int8,
        )
        parents = replace(
            graph, target, MXINT8ToFP32DequantOp(), [codes], [len(shape) - 1]
        )
        discard(graph, parents)
        count += 1
    if not count:
        raise ValueError("Laya graph contains no LayerNorm expressions")

    # Quantization after an MXINT8 decode is idempotent. Keep only the row
    # reshape of the codes, so matrix consumers never materialize FP32 here.
    for quant in list(graph.body):
        if not isinstance(quant, FP32ToMXINT8QuantOp):
            continue
        source = graph.node_table[quant.args[0]]
        while isinstance(source, (CloneOp, ViewOp, PermuteOp)):
            parent = graph.node_table[source.args[0]]
            if isinstance(source, PermuteOp):
                shape = parent.tensor_meta["shape"]
                axes = source.args[1]
                # Moving singleton dimensions preserves the packed row order.
                if axes[-1] != len(shape) - 1 or [
                    axis for axis in axes if shape[axis] != 1
                ] != [axis for axis, size in enumerate(shape) if size != 1]:
                    break
            source = parent
        if not isinstance(source, MXINT8ToFP32DequantOp):
            continue
        codes = graph.node_table[source.args[0]]
        if not isinstance(codes, FixedNormInt8Op):
            continue
        if quant.args[1] != len(quant.tensor_meta["shape"]) - 1:
            raise ValueError(
                f"{quant.name}: normalized MXINT8 rows must keep their last axis"
            )
        parents = replace(
            graph, quant, ViewOp(), [codes], [list(quant.tensor_meta["shape"])]
        )
        discard(graph, parents)
    sort_graph(graph)
    print(f"Mapped {count} LayerNorm expressions to integer normalization")
    return [values[node.name] for node in graph.params]
