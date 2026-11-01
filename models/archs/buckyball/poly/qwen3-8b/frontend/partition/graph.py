from buddy.compiler.graph.operation import (
    BatchMatmulOp,
    CloneOp,
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
    """The decoder across poly: the reduce core and a rank on each of its
    poly cores (as many as design.pb has), with the attention heads, the FFN
    channels and the vocabulary split across the ranks. Each rank owns the
    KV cache of its heads; every core computes the positions, the mask and
    the rotary tables it uses."""
    (embedding,) = [node for node in graph.body if isinstance(node, EmbeddingOp)]
    hidden, frontier = {embedding.name}, [embedding]
    while frontier:
        for child in frontier.pop().children:
            if child not in hidden:
                hidden.add(child)
                frontier.append(graph.node_table[child])
    replicated = {
        node.name
        for node in graph.body
        if node.name not in hidden and not isinstance(node, PlaceholderOp)
    }
    # The inputs are the tokens, the start position, then the keys and the
    # values of every layer: [1, heads, capacity, head_dim].
    states = {node.name: 1 for node in graph.inputs[2:]}
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
    ranks = sum(core.component == "poly" for core in design.device.cores)
    return partition(graph, params, ranks, states, replicated, columns, {}, classify)


def classify(device, segment):
    """The semantic graphs of the decoder. On each rank: `attention`, the
    attention of a layer over its heads (q, k, v, their norms and RoPE, its
    slice of the KV cache, scores, softmax, context, its K piece of o_proj);
    `mlp`, the MLP of a layer over its channels (gate, up, SiLU·up, its K
    piece of down); `lm_head`, its piece of the vocabulary. On the reduce
    core: `embed`, the embedding and the start of the first attention (its
    norm, the quantized inputs of q, k, v); `attention_out`, the end of an
    attention (the partial sums of the ranks in rank order, the residual)
    and the start of the MLP (its norm, the quantized inputs of gate and
    up); `mlp_out`, the end of an MLP and the start of the next attention;
    `final`, the end of the last MLP, the final norm and the input of the
    LM head; `cache`, the KV cache of a layer joined from the ranks'
    slices; `logits`, the logits joined from the ranks' pieces."""

    def head(segment):
        return any(
            name.rsplit("_r", 1)[0] == device.output.args[0] for name in segment.outputs
        )

    ops = [node for _, node in segment.ops]
    if segment.core_class == "poly":
        if any(isinstance(node, BatchMatmulOp) for node in ops):
            return "attention"
        return "lm_head" if head(segment) else "mlp"
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
    return f"{block}_out"
