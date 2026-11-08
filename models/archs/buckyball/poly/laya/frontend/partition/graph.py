from buddy.compiler.graph.operation import (
    AbsOp,
    BatchMatmulOp,
    CloneOp,
    CosOp,
    EmbeddingOp,
    PermuteOp,
    PlaceholderOp,
    RsqrtOp,
    TransposeMatmulFusedOp,
    ViewOp,
)
from compiler.frontend.partition import partition
from compiler.frontend.quant import FP32ToMXINT8QuantOp, MXINT8ToFP32DequantOp


def cut(graph, params, design):
    """Laya across poly: the reduce core and a rank on each of its poly
    cores (as many as design.pb has), with the attention heads and the FFN
    channels of the encoder and the head split across the ranks. The fused
    Linears (q, k, v of the encoder and of the head, gate and up of the
    GeGLU) split each of their parts on its own; the GeGLU channels, 82
    MXINT8 blocks, are dealt out, the first ranks one more when they do not
    divide evenly (with 8 ranks, 11 to the first two and 10 to the others).
    The reduce core computes the action head whole: it reads the marker
    features, not a normalized activation. Every core computes the
    positions, the masks and the rotary tables it uses."""
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
        frontier = [node]
        for _ in range(3):
            frontier = [graph.node_table[name] for n in frontier for name in n.parents]
            if any(isinstance(n, RsqrtOp) for n in frontier):
                columns.add(linear.name)
                break
    # q, k, v in one Linear of 3 x 1024 outputs; gate and up in one of
    # 2 x 2624.
    segments = {}
    for node in graph.body:
        if node.name not in columns:
            continue
        width = node.tensor_meta["shape"][-1]
        if width == 3 * 1024:
            segments[node.name] = 3
        elif width == 2 * 2624:
            segments[node.name] = 2
    ranks = sum(core.component == "poly" for core in design.device.cores)
    return partition(graph, params, ranks, {}, replicated, columns, segments, classify)


def classify(device, segment):
    """The semantic graphs of Laya. On each rank, for a layer of the
    ModernBERT encoder: `attention_global` and `attention_local` (the
    sliding window), the attention with RoPE over its heads and its K piece
    of Wo; `mlp`, the GeGLU over its channels and its K piece of Wo. For a
    layer of the head: `head_attention`, the attention with biases over its
    heads; `head_mlp`, the feed-forward over its channels. `scorer`, the
    first Linear of the scorer over its channels and its K piece of the
    second. On the reduce core: `embed`, the embeddings and the start of the
    first layer; `attention_out` and `mlp_out`, the end of a block of the
    encoder (the partial sums of the ranks in rank order, the residual) and
    the start of the next (its norm, its quantized input);
    `head_attention_out` and `head_mlp_out`, the same in the head;
    `head_in`, the end of the encoder (its final norm), the type embedding
    and the start of the head; `head_out`, the end of the head, the marker
    features and the norm of the scorer; `actions`, the end of the scorer,
    the scores and the action head."""

    def scorer(segment):
        return segment.core_class == "poly" and any(
            name in device.output.args
            for reader in segment.readers
            for name in reader.outputs
        )

    ops = [node for _, node in segment.ops]
    head = any(s.graph_class == "head_attention" for s in device.segments)
    if segment.core_class == "poly":
        if any(isinstance(node, BatchMatmulOp) for node in ops):
            if not any(isinstance(node, CosOp) for node in ops):
                return "head_attention"
            return (
                "attention_local"
                if any(isinstance(node, AbsOp) for node in ops)
                else "attention_global"
            )
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
        if name in device.owner and device.owner[name].core_class == "poly"
    }
    return f"{block}_out"
