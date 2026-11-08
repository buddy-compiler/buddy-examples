from buddy.compiler.graph.operation import (
    AddOp,
    ClampOp,
    CloneOp,
    Conv2dOp,
    DivOp,
    ErfOp,
    ExpOp,
    MatmulOp,
    MulOp,
    NegOp,
    PermuteOp,
    ReluOp,
    SigmoidOp,
    SiluOp,
    SubOp,
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
    ErfOp,
    ExpOp,
    MulOp,
    NegOp,
    ReluOp,
    SigmoidOp,
    SiluOp,
    SubOp,
)


def nwc(shape):
    batch, channels, width = shape
    return [batch, width, channels]


def tensor_args(graph, node):
    names = []
    for arg in node.args:
        for name in arg if isinstance(arg, (list, tuple)) else [arg]:
            if isinstance(name, str) and name in graph.node_table:
                names.append(name)
    return names


def transpose(graph, source, layout, users, index):
    """A permute of `source` to `layout` (NWC or NCW, both [0, 2, 1]) read by
    `users` instead of `source`, placed at `index` of the body."""
    node = PermuteOp()
    node.name = f"{source.name}_{layout}"
    node.add_argument(source.name)
    node.add_argument([0, 2, 1])
    shape = list(source.tensor_meta["shape"])
    node.tensor_meta = {
        "shape": [shape[0], shape[2], shape[1]],
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
    return node


def permute(graph):
    """The 1-D convolutions and the layers around them run in NWC; the
    activations turn NWC once where the region starts and back to NCW where
    it ends. x @ W.T of a Linear becomes one matmul with a transposed right
    operand. Every weight keeps its checkpoint layout."""
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

    # The region: 1-D convolutions, and the elementwise layers whose 3-D
    # inputs all are in the region.
    region = set()
    for node in graph.body:
        inputs = tensor_args(graph, node)
        inside = bool(inputs) and all(name in region for name in inputs)
        if isinstance(node, Conv2dOp):
            region.add(node.name)
        elif isinstance(node, elementwise):
            if inside and len(node.tensor_meta["shape"]) == 3:
                region.add(node.name)

    for name in region:
        node = graph.node_table[name]
        node.tensor_meta["shape"] = nwc(node.tensor_meta["shape"])
        if isinstance(node, Conv2dOp):
            node._layout = "NWC_FCW"

    # The boundaries: NCW to NWC into a convolution, NWC to NCW out of the
    # region.
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
            transpose(graph, source, "nwc", users, first)
    for name in [node.name for node in graph.body if node.name in region]:
        source = graph.node_table[name]
        users = [user for user in source.children if user not in region]
        if not users:
            continue
        ncw = transpose(graph, source, "ncw", users, graph.body.index(source) + 1)
        # A permute back to NWC right after cancels with it: its users read
        # the region value.
        for name in list(ncw.children):
            back = graph.node_table[name]
            if not isinstance(back, PermuteOp) or list(back.args[1]) != [0, 2, 1]:
                continue
            for user in back.children:
                user = graph.node_table[user]
                user._arguments = [
                    (
                        [source.name if x == back.name else x for x in arg]
                        if isinstance(arg, (list, tuple))
                        else (source.name if arg == back.name else arg)
                    )
                    for arg in user._arguments
                ]
                user._parents = [
                    source.name if x == back.name else x for x in user._parents
                ]
                source.add_children(user.name)
            graph.delete_node(back, [ncw])
        if not ncw.children:
            graph.delete_node(ncw, [source])
