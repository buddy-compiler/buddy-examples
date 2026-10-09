from buddy_mlir import ir
from buddy_mlir.dialects import tosa
from buddy.compiler.ops import tosa as ops


def reorder(weight):
    if weight.ndim == 4:
        # OIHW -> HWIO -> [H*W*I, O], without offline padding.
        return weight.transpose(2, 3, 1, 0).reshape(-1, weight.shape[0]).copy()
    if weight.ndim == 2:
        return weight.T.copy()
    raise ValueError("Toy AlexNet expects convolution and linear weight tensors")


def pool_result(node, symbols):
    pooled = ops.maxpool2d_op(node, symbols).result
    # Restore NCHW so the following convolution and flatten retain Torch order.
    result = tosa.TransposeOp(
        ir.RankedTensorType.get(node.tensor_meta["shape"], ir.F32Type.get()),
        pooled,
        ops._create_permutation_attr([0, 3, 1, 2]),
    ).result
    return result
