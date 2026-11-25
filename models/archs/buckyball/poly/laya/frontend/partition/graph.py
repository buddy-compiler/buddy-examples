from buddy.compiler.graph import NodeType
from buddy.compiler.graph.operation import (
    AbsOp,
    AddOp,
    BatchMatmulOp,
    CloneOp,
    CosOp,
    SinOp,
    WhereOp,
    EmbeddingOp,
    GetItemOp,
    PermuteOp,
    PlaceholderOp,
    RsqrtOp,
    TransposeMatmulFusedOp,
    ViewOp,
)
from buddy.compiler.graph.transform.quantization.quantize import sort_graph
from compiler.frontend.partition import partition
from compiler.frontend.quant import FP32ToMXINT8QuantOp, MXINT8ToFP32DequantOp
from frontend.norm import FixedNormInt8Op


def unfuse(graph, params):
    """Each fused q, k, v Linear (3 x 1024 outputs, with or without a bias)
    as three Linears over the same quantized input, one for each of q, k and
    v: a rank then writes its heads of each in a buffer of its own, which a
    thread reads whole, instead of a strided view into the fused output. The
    packed weight and the bias split along their output rows; every value is
    the same."""
    values = dict(zip((node.name for node in graph.params), params, strict=True))

    def placeholder(name, value, dtype):
        node = PlaceholderOp()
        node.name = name
        node.tensor_meta = {"shape": list(value.shape), "dtype": dtype}
        graph.add_node(node, NodeType.FakeNode)
        values[name] = value
        return node

    def operation(op, name, arguments, shape, dtype):
        node = op()
        node.name = name
        for argument in arguments:
            node.add_argument(argument)
            if isinstance(argument, str):
                node.add_parent(argument)
                graph.node_table[argument].add_children(name)
        node.tensor_meta = {"shape": list(shape), "dtype": dtype}
        graph.add_node(node)
        return node

    for linear in list(graph.body):
        if (
            not isinstance(linear, TransposeMatmulFusedOp)
            or linear.tensor_meta["shape"][-1] != 3 * 1024
        ):
            continue
        restored = graph.node_table[linear.args[1]]
        weight = graph.node_table[restored.args[0]]
        # The fused output, its bias, its views down to the pieces q, k and v
        # of it (the items of an unbind or the selections of a copy).
        (name,) = linear.children
        added = graph.node_table[name]
        bias = added.args[0] if isinstance(added, AddOp) else None
        chain = [added] if bias else []
        node = graph.node_table[(added if bias else linear).children[0]]
        while len(node.children) == 1:
            chain.append(node)
            node = graph.node_table[node.children[0]]
        chain.append(node)
        # The token count, as the first view names it.
        view = next(n for n in chain if isinstance(n, ViewOp))
        size = view.parents[1]
        for piece in [graph.node_table[name] for name in node.children]:
            index = piece.args[1] if isinstance(piece, GetItemOp) else piece.args[2]
            rows = slice(index * 1024, (index + 1) * 1024)
            part = placeholder(
                f"{weight.name}_qkv{index}",
                values[weight.name][rows],
                weight.tensor_meta["dtype"],
            )
            part = operation(
                type(restored),
                f"{restored.name}_qkv{index}",
                [part.name, *restored.args[1:]],
                [1024, *restored.tensor_meta["shape"][1:]],
                restored.tensor_meta["dtype"],
            )
            shape = [*linear.tensor_meta["shape"][:-1], 1024]
            value = operation(
                TransposeMatmulFusedOp,
                f"{linear.name}_qkv{index}",
                [linear.args[0], part.name, *linear.args[2:]],
                shape,
                linear.tensor_meta["dtype"],
            )
            if bias:
                part = placeholder(
                    f"{bias}_qkv{index}",
                    values[bias][rows],
                    graph.node_table[bias].tensor_meta["dtype"],
                )
                value = operation(
                    AddOp,
                    f"{added.name}_qkv{index}",
                    [part.name, value.name],
                    shape,
                    added.tensor_meta["dtype"],
                )
            # The piece itself: the same values, row after row.
            dimensions = [
                size if not isinstance(d, int) else d
                for d in piece.tensor_meta["shape"]
            ]
            result = operation(
                ViewOp,
                f"{piece.name}_unfused",
                [value.name, dimensions],
                piece.tensor_meta["shape"],
                piece.tensor_meta["dtype"],
            )
            result.add_parent(size)
            graph.node_table[size].add_children(result.name)
            for name in piece.children:
                user = graph.node_table[name]
                user.args[:] = [
                    result.name if a == piece.name else a for a in user.args
                ]
                user.parents[:] = [
                    result.name if p == piece.name else p for p in user.parents
                ]
                result.add_children(name)
            piece.children.clear()
            graph.delete_node(piece, [graph.node_table[n] for n in piece.parents])
        for node in reversed([linear, *chain]):
            graph.delete_node(node, [graph.node_table[n] for n in node.parents])
        for node in (restored, weight, *([graph.node_table[bias]] if bias else [])):
            if not node.children:
                graph.delete_node(node, [graph.node_table[n] for n in node.parents])
    sort_graph(graph)
    return [values[node.name] for node in graph.params]


def cut(graph, params, design):
    """Laya across poly: the reduce hart and a rank on each of its poly
    harts (as many as design.pb has), with the attention heads and the FFN
    channels of the encoder and the head split across the ranks. q, k and v
    are three Linears (unfuse); the fused gate and up of the GeGLU split
    each of their parts on its own; the GeGLU channels, 82
    MXINT8 blocks, are dealt out, the first ranks one more when they do not
    divide evenly (with 8 ranks, 11 to the first two and 10 to the others).
    The reduce hart computes the action head whole: it reads the marker
    features, not a normalized activation. Masks and rotary tables are
    computed once on each consuming hart."""
    hidden = set()
    frontier = [node for node in graph.body if isinstance(node, EmbeddingOp)]
    while frontier:
        node = frontier.pop()
        for child in node.children:
            if child not in hidden:
                hidden.add(child)
                frontier.append(graph.node_table[child])
    replicated = {
        node.name
        for node in graph.body
        if node.name not in hidden and not isinstance(node, PlaceholderOp)
    }
    # The Linears that read a normalized activation (q, k, v, gate, up and
    # the LM head) split along their output channels; the Linears after them
    # read their pieces and split along K.
    columns = set()
    for linear in graph.body:
        if not isinstance(linear, TransposeMatmulFusedOp) or linear.name in replicated:
            continue
        node = graph.node_table[linear.args[0]]
        while isinstance(
            node,
            (ViewOp, CloneOp, PermuteOp, FP32ToMXINT8QuantOp, MXINT8ToFP32DequantOp),
        ):
            node = graph.node_table[node.args[0]]
        # A normalization ends in its scale (and shift) after the reciprocal
        # square root.
        if isinstance(node, FixedNormInt8Op):
            columns.add(linear.name)
            continue
        frontier = [node]
        for _ in range(3):
            frontier = [graph.node_table[name] for n in frontier for name in n.parents]
            if any(isinstance(n, RsqrtOp) for n in frontier):
                columns.add(linear.name)
                break
    # Gate and up in one Linear of 2 x 2624 outputs (unfuse made q, k and v
    # three).
    segments = {
        node.name: 2
        for node in graph.body
        if node.name in columns and node.tensor_meta["shape"][-1] == 2 * 2624
    }
    ranks = sum(hart.component == "poly" for hart in design.device.harts)
    preparation = {
        node.name
        for node in graph.body
        if node.name in replicated and isinstance(node, (CosOp, SinOp, WhereOp))
    }
    # Keep the complete mask and rotary preparation together on each hart.
    frontier = list(preparation)
    while frontier:
        node = graph.node_table[frontier.pop()]
        for name in [*node.parents, *node.children]:
            if name in replicated and name not in preparation:
                preparation.add(name)
                frontier.append(name)
    (reducer,) = [
        hart.component
        for hart in design.device.harts
        if hart.hart_id == design.workload.control_hart
    ]
    return partition(
        graph,
        params,
        ranks,
        {},
        replicated,
        columns,
        segments,
        preparation,
        set(),
        native=set(),
        classify=lambda device, segment: classify(device, segment, preparation),
        quantized_partials=set(),
        units={},
        classes=[reducer] + ["poly"] * ranks,
        sink=False,
    )


def classify(device, segment, preparation):
    """The semantic graphs of Laya. On each rank, for a layer of the
    ModernBERT encoder: `attention_global` and `attention_local` (the
    sliding window), the attention with RoPE over its heads and its K piece
    of Wo; `mlp`, the GeGLU over its channels and its K piece of Wo. For a
    layer of the head: `head_attention`, the attention with biases over its
    heads; `head_mlp`, the feed-forward over its channels. `scorer`, the
    first Linear of the scorer over its channels and its K piece of the
    second. On the reduce hart: `embed`, the embeddings and the start of the
    first layer; `attention_out` and `mlp_out`, the end of a block of the
    encoder (the partial sums of the ranks in rank order, the residual) and
    the start of the next (its norm, its quantized input);
    `head_attention_out` and `head_mlp_out`, the same in the head;
    `head_in`, the end of the encoder (its final norm), the type embedding
    and the start of the head; `head_out`, the end of the head, the marker
    features and the norm of the scorer; `actions`, the end of the scorer,
    the scores and the action head."""

    def scorer(segment):
        return segment.hart_class == "poly" and any(
            name in device.output.args
            for reader in segment.readers
            for name in reader.outputs
        )

    if segment.hart_class == "poly" and any(
        node.name in preparation or "_local_r" in node.name for _, node in segment.ops
    ):
        return "prepare"
    ops = [node for _, node in segment.ops]
    supplied = []
    seen = set()
    frontier = list(segment.inputs)
    while frontier:
        name = frontier.pop()
        if name in seen:
            continue
        seen.add(name)
        if "_local_r" not in name and (
            name not in device.owner or device.owner[name].graph_class != "prepare"
        ):
            continue
        node = device.nodes[name]
        supplied.append(node)
        frontier.extend(node.parents)
    rotary = any(isinstance(node, CosOp) for node in [*ops, *supplied])
    local_mask = any(isinstance(node, AbsOp) for node in [*ops, *supplied])
    head = any(s.graph_class == "head_attention" for s in device.segments)
    if segment.hart_class == "poly":
        if any(isinstance(node, BatchMatmulOp) for node in ops):
            if not rotary:
                return "head_attention"
            return "attention_local" if local_mask else "attention_global"
        if scorer(segment):
            return "scorer"
        return "head_mlp" if head else "mlp"
    if any(name in device.output.args for name in segment.outputs):
        return "actions"
    if any(isinstance(node, EmbeddingOp) for node in ops):
        return (
            "head_in"
            if any(name in device.owner for name in segment.inputs)
            else "embed"
        )
    if any(scorer(reader) for reader in segment.readers):
        return "head_out"
    (block,) = {
        device.owner[name].graph_class.removesuffix("_global").removesuffix("_local")
        for name in segment.inputs
        if name in device.owner
        and device.owner[name].hart_class == "poly"
        and device.owner[name].graph_class != "prepare"
    }
    return f"{block}_out"
