import torch
import buddy_mlir.ir as ir
from buddy.compiler.graph.operation import ErfOp
from torch.overrides import TorchFunctionMode


class ApproxErfOp(ErfOp):
    pass


def approximate_erf_op(node, symbol_table):
    value = symbol_table[(node.args[0], 0)]
    return ir.Operation.create(
        "math.erf",
        results=[value.type],
        operands=[value],
        attributes={"fastmath": ir.Attribute.parse("#arith.fastmath<afn>")},
    ).result


ops_registry = {"ApproxErfOp": approximate_erf_op}


def approximate_erf(graph):
    count = 0
    for node in list(graph.body):
        if isinstance(node, ErfOp):
            replacement = ApproxErfOp()
            replacement.name = node.name
            graph.displace_node(node, replacement)
            count += 1
    if not count:
        raise ValueError("Laya graph contains no erf activations")
    print(f"Mapped {count} activations to FP32 polynomial approximation")


def erf(value):
    # The FP32 coefficients and FMA order of MLIR's ErfPolynomialApproximation.
    p = torch.tensor(
        [
            [
                0,
                1.12837916222975858,
                -0.523018562988006470,
                0.209741709609267072,
                0.0258146801602987875,
            ],
            [
                0,
                1.12750687816789140,
                -0.364721408487825775,
                0.118407396425136952,
                0.0370645533056476558,
            ],
            [
                -0.00330093071049483172,
                0.00351961938357697011,
                -0.00141373622814988039,
                0.000253447094961941348,
                -0.0000171048029455037401,
            ],
        ],
        dtype=torch.float32,
        device=value.device,
    )
    q = torch.tensor(
        [
            [
                1,
                -0.4635138185962547255,
                0.5192301327279782447,
                -0.1318089722204810087,
                0.07397964654672315005,
            ],
            [
                1,
                -0.327607011824493086,
                0.448369090658821977,
                -0.0883462621207857930,
                0.0572442770283176093,
            ],
            [
                1,
                -2.06069165953913769,
                1.62705939945477759,
                -0.583389859211130017,
                0.0821908939856640930,
            ],
        ],
        dtype=torch.float32,
        device=value.device,
    )
    x = value.abs()
    region = torch.where(x < 0.8, 0, torch.where(x < 2, 1, 2))
    wide = x.to(torch.float64)

    def polynomial(table):
        result = table[region, 4]
        for index in range(3, -1, -1):
            result = (
                wide * result.to(torch.float64) + table[region, index].to(torch.float64)
            ).to(torch.float32)
        return result

    result = (region == 2).to(torch.float32) + polynomial(p) / polynomial(q)
    result = torch.where(x < 3.75, result, 1)
    return torch.where(value < 0, -result, result)


class Erfs(TorchFunctionMode):
    calls = 0

    def __torch_function__(self, function, types, args=(), kwargs=None):
        if getattr(function, "__name__", None) == "gelu":
            type(self).calls += 1
            x = args[0]
            return (x * 0.5) * (1 + erf(x * 0.7071067811865476))
        return function(*args, **(kwargs or {}))
