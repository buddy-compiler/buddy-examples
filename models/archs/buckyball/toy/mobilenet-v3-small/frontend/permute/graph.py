from buddy.compiler.graph.operation import (
    AdaptiveAvgPool2dOp,
    AddOp,
    CatOp,
    ClampOp,
    CloneOp,
    Conv2dOp,
    DivOp,
    ExpOp,
    GetItemOp,
    MatmulOp,
    MaxPool2dOp,
    MeanOp,
    MulOp,
    NativeBatchNormLegitNoTrainingOp,
    NegOp,
    PermuteOp,
    ReluOp,
    SigmoidOp,
    SiluOp,
    SplitOp,
    SplitWithSizesOp,
    SubOp,
    TOp,
)
from buddy.compiler.graph.transform.fuse_ops import (
    decompose_addmm_to_mm_add,
    transpose_matmul_fusion,
)

elementwise = (
    AddOp,
    ClampOp,
    CloneOp,
    DivOp,
    ExpOp,
    MulOp,
    NegOp,
    ReluOp,
    SigmoidOp,
    SiluOp,
    SubOp,
)


def nhwc(shape):
    batch, channels, height, width = shape
    return [batch, height, width, channels]


def tensor_args(graph, node):
    names = []
    for arg in node.args:
        for name in arg if isinstance(arg, (list, tuple)) else [arg]:
            if isinstance(name, str) and name in graph.node_table:
                names.append(name)
    return names


def rank(graph, name):
    return len(graph.node_table[name].tensor_meta["shape"])


def transpose(graph, source, perm, users, index):
    """A permute of `source` read by `users` instead of `source`, placed at
    `index` of the body."""
    node = PermuteOp()
    node.name = f"{source.name}_{'nhwc' if perm == [0, 2, 3, 1] else 'nchw'}"
    node.add_argument(source.name)
    node.add_argument(perm)
    shape = list(source.tensor_meta["shape"])
    node.tensor_meta = {
        "shape": [shape[i] for i in perm],
        "dtype": source.tensor_meta["dtype"],
    }
    node.add_parent(source.name)
    graph.add_node(node)
    graph.body.remove(node)
    graph.body.insert(index, node)
    # The inputs and parameters at or after `index` move one place down.
    graph._inputs = [i + (i >= index) for i in graph._inputs]
    graph._fake_params = [i + (i >= index) for i in graph._fake_params]
    for user in users:
        user = graph.node_table[user]
        user._arguments = [
            (
                [node.name if x == source.name else x for x in arg]
                if isinstance(arg, (list, tuple))
                else (node.name if arg == source.name else arg)
            )
            for arg in user._arguments
        ]
        user._parents = [node.name if x == source.name else x for x in user._parents]
        source._children.remove(user.name)
        node.add_children(user.name)
    source.add_children(node.name)


def permute(graph):
    """The convolutions and the layers around them run in NHWC, the layout
    TOSA convolves and pools in; the activations turn NHWC once where the
    region starts and back to NCHW where it ends. x @ W.T of a Linear becomes
    one matmul with a transposed right operand. Every weight keeps its
    checkpoint layout."""
    for t in [node for node in graph.body if isinstance(node, TOp)]:
        permute = PermuteOp()
        permute.name = t.name
        graph.displace_node(t, permute)
        permute.add_argument([1, 0])
    decompose_addmm_to_mm_add(graph)
    for matmul in list(graph.body):
        if not isinstance(matmul, MatmulOp):
            continue
        weight = graph.node_table[matmul.args[1]]
        if not isinstance(weight, PermuteOp) or list(weight.args[1]) != [1, 0]:
            continue
        parents = [graph.node_table[name] for name in matmul.parents]
        transpose_matmul_fusion(
            graph, matmul, weight, parents, "transpose_matmul_fusion"
        )

    # The region: convolutions, and the layers whose 4-D inputs all are in
    # the region and whose channel axis can move last.
    region = set()
    for node in graph.body:
        inputs = tensor_args(graph, node)
        inside = bool(inputs) and all(name in region for name in inputs)
        if isinstance(node, Conv2dOp):
            region.add(node.name)
        elif isinstance(
            node, (MaxPool2dOp, AdaptiveAvgPool2dOp, NativeBatchNormLegitNoTrainingOp)
        ):
            if node.args[0] in region:
                region.add(node.name)
        elif isinstance(node, GetItemOp):
            # A piece of a split, or the output of a batch norm.
            source = graph.node_table[node.args[0]]
            if node.args[0] in region and (
                isinstance(source, (SplitOp, SplitWithSizesOp)) or node.args[1] == 0
            ):
                region.add(node.name)
        elif isinstance(node, elementwise):
            if inside and len(node.tensor_meta["shape"]) == 4:
                region.add(node.name)
        elif isinstance(node, MeanOp):
            dims = {d % 4 for d in node.args[1]}
            if inside and dims == {2, 3} and node.args[2]:
                region.add(node.name)
        elif isinstance(node, CatOp):
            if inside and node.args[1] % 4 == 1:
                region.add(node.name)
        elif isinstance(node, (SplitOp, SplitWithSizesOp)):
            if inside and node.args[2] % 4 == 1:
                region.add(node.name)

    for name in region:
        node = graph.node_table[name]
        shape = node.tensor_meta["shape"]
        if isinstance(
            node, (NativeBatchNormLegitNoTrainingOp, SplitOp, SplitWithSizesOp)
        ):
            node.tensor_meta["shape"] = [
                nhwc(s) if len(s) == 4 else list(s) for s in shape
            ]
        else:
            node.tensor_meta["shape"] = nhwc(shape)
        if isinstance(node, Conv2dOp):
            node._layout = "NHWC_FCHW"
        elif isinstance(
            node, (MaxPool2dOp, AdaptiveAvgPool2dOp, NativeBatchNormLegitNoTrainingOp)
        ):
            node._layout = "NHWC"
        elif isinstance(node, MeanOp):
            node._arguments[1] = [1, 2]
        elif isinstance(node, CatOp):
            node._arguments[1] = 3
        elif isinstance(node, (SplitOp, SplitWithSizesOp)):
            node._arguments[2] = 3

    # The boundaries: NCHW to NHWC into a convolution, NHWC to NCHW out of
    # the region.
    sources = dict.fromkeys(
        node.args[0]
        for node in graph.body
        if isinstance(node, Conv2dOp) and node.args[0] not in region
    )
    for name in sources:
        source = graph.node_table[name]
        users = [user for user in source.children if user in region]
        if users:
            # Placed before its first user: the placeholders must stay
            # together, they take the function arguments in order.
            first = min(graph.body.index(graph.node_table[user]) for user in users)
            transpose(graph, source, [0, 2, 3, 1], users, first)
    for name in [node.name for node in graph.body if node.name in region]:
        source = graph.node_table[name]
        if isinstance(
            source, (NativeBatchNormLegitNoTrainingOp, SplitOp, SplitWithSizesOp)
        ):
            continue
        users = [user for user in source.children if user not in region]
        if users:
            transpose(graph, source, [0, 3, 1, 2], users, graph.body.index(source) + 1)
