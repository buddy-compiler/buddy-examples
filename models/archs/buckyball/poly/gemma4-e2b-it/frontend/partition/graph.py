from buddy.compiler.graph.operation import (
    BatchMatmulOp,
    CloneOp,
    EmbeddingOp,
    IndexPutOp,
    PermuteOp,
    PlaceholderOp,
    PowOp,
    RsqrtOp,
    SliceOp,
    TransposeMatmulFusedOp,
    ViewOp,
)
from compiler.frontend.partition import partition
from compiler.frontend.quant import FP32ToMXINT8QuantOp, MXINT8ToFP32DequantOp


def cut(graph, params, design):
    """The decoder across poly: the reduce core and a rank on each of its
    poly cores (as many as design.pb has), with the query heads, the FFN
    channels and the vocabulary split across the ranks. Gemma 4 E2B has one
    KV head: every poly core computes the keys and values and keeps the
    whole KV cache, and reads the head of its query. The reduce core computes the per-layer inputs of all layers whole:
    their projection reads the embedding, not a normalized activation.
    Every core computes the positions, the masks and the rotary tables it
    uses."""
    hidden = set()
    frontier = [node for node in graph.body if isinstance(node, EmbeddingOp)]
    while frontier:
        node = frontier.pop()
        for child in node.children:
            if child not in hidden:
                hidden.add(child)
                frontier.append(graph.node_table[child])
    # The key and value projections: the Linears the cache writes come from.
    projections, frontier = set(), [
        graph.node_table[node.args[2]]
        for node in graph.body
        if isinstance(node, IndexPutOp)
    ]
    while frontier:
        node = frontier.pop()
        if isinstance(node, TransposeMatmulFusedOp):
            projections.add(node)
            continue
        frontier.extend(graph.node_table[name] for name in node.parents)
    # Their inputs and weights, and all that follows from them up to the
    # attention products.
    keys = {name for node in projections for name in node.args[:2]}
    frontier = list(projections)
    while frontier:
        node = frontier.pop()
        keys.add(node.name)
        for child in node.children:
            user = graph.node_table[child]
            if (
                child not in keys
                and not isinstance(user, BatchMatmulOp)
                and child != "output"
            ):
                frontier.append(user)
    replicated = keys | {
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
        # square root, which Gemma 4 computes as a power of -0.5.
        frontier = [node]
        for _ in range(3):
            frontier = [graph.node_table[name] for n in frontier for name in n.parents]
            if any(
                isinstance(n, RsqrtOp) or (isinstance(n, PowOp) and n.args[1] == -0.5)
                for n in frontier
            ):
                columns.add(linear.name)
                break
    ranks = sum(core.component == "poly" for core in design.device.cores)
    return partition(graph, params, ranks, {}, replicated, columns, {}, classify)


def classify(device, segment):
    """The semantic graphs of Gemma 4. On each rank: `attention`, the
    attention of a layer over its query heads, with the keys and values it
    computes itself (a graph per kind of layer: sliding or full, with its
    own KV or reading that of an earlier layer); `mlp`, the MLP of a layer
    over its channels (the layers that share KV have a wider one);
    `per_layer`, the gate of the per-layer input of a layer over its
    channels, which reads its piece of that input; `lm_head`, its piece of
    the vocabulary. On the reduce core: `embed`, the embedding, the
    per-layer inputs of all layers dealt out to the ranks and the start of
    the first attention; after each block of a layer the partial sums of
    the ranks in rank order, the norms and the residual of the block and the
    start of the next block: `attention_out`, `mlp_out` and `layer_out`
    (with the layer scale; the next attention reads its own keys and values
    or not); `final`, the end of the last layer, the final norm and the
    input of the LM head; `cache`, the keys and values of a layer that has
    its own, for the program's outputs (sliding or full); `logits`, the
    logits joined from the ranks' pieces."""

    def head(segment):
        return any(
            name.rsplit("_r", 1)[0] == device.output.args[0] for name in segment.outputs
        )

    ops = [node for _, node in segment.ops]
    if segment.core_class == "poly":
        if any(isinstance(node, BatchMatmulOp) for node in ops):
            return "attention"
        if head(segment):
            return "lm_head"
        if any(isinstance(device.nodes.get(name), SliceOp) for name in segment.inputs):
            return "per_layer"
        return "mlp"
    if segment.join:
        return "logits" if device.output.args[0] in segment.outputs else "cache"
    if any(isinstance(node, EmbeddingOp) for node in ops):
        return "embed"
    if any(head(reader) for reader in segment.readers):
        return "final"
    (block,) = {
        device.owner[name].graph_class
        for name in segment.inputs
        if name in device.owner and device.owner[name].core_class == "poly"
    }
    return f"{block}_out" if block != "per_layer" else "layer_out"
