from buddy.compiler.graph.operation import (
    BatchMatmulOp,
    CloneOp,
    PermuteOp,
    PlaceholderOp,
    RsqrtOp,
    SliceOp,
    ViewOp,
)
from compiler.frontend.partition import partition, restoring, separate_biases
from compiler.frontend.quant import FP32ToMXINT8QuantOp, MXINT8ToFP32DequantOp


def ranks(design):
    """The hart class of the reduce hart and of every rank: the action hart
    reduces and runs the first rank, the dense harts the others, in the
    order of design.pb."""
    harts = [hart.component for hart in design.device.harts]
    ranks = [c for c in harts if c == "action"] + [c for c in harts if c == "dense"]
    return [ranks[0]] + ranks


def cut(graph, params, design, name, example, prepared, heads, groups, head_dim):
    """The graph `name` (vision, prefix or step) across the piplus harts: the
    attention heads, the MLP channels split across the ranks, each rank the
    keys and values of its key heads (`heads` of them, each shared by
    `groups` query heads of `head_dim` values): the projections of q (and
    its rotated copy) deal out the query heads of a key head at a time, k
    and v a key head. The ranks of 5 key heads over 4 harts take 2, 1, 1, 1.
    Every hart computes what it reads of the inputs `prepared` (indices of
    `example`: the masks and the rotary angles) itself."""
    separate_biases(graph)
    classes = ranks(design)
    count = len(classes) - 1
    # The keys and values: [1, tokens, heads, head_dim].
    states = {
        node.name: 2
        for node in graph.inputs
        if list(node.tensor_meta["shape"])[2:] == [heads, head_dim]
    }
    # The values computed from the other inputs (the activations, the
    # states); the rest every hart computes itself. Dynamo orders the inputs
    # of the graph as the forward first reads them.
    order = [
        next(i for i, value in enumerate(example) if value is input_)
        for input_ in graph._runtime_inputs_ref
    ]
    activations = {
        node.name
        for node, index in zip(graph.inputs, order, strict=True)
        if index not in prepared
    }
    hidden, frontier = set(activations), [graph.node_table[n] for n in activations]
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
    # The Linears that read a normalized activation (q, its rotated copy, k,
    # v, gate, up) split along their output channels, those of q a key
    # head's query heads at a time, k and v a key head; the Linears after
    # them read their pieces and split along K.
    columns, units = set(), {}
    for linear in graph.body:
        if not restoring(graph, linear) or linear.name in replicated:
            continue
        node = graph.node_table[linear.args[0]]
        while isinstance(
            node,
            (
                ViewOp,
                SliceOp,
                CloneOp,
                PermuteOp,
                FP32ToMXINT8QuantOp,
                MXINT8ToFP32DequantOp,
            ),
        ):
            node = graph.node_table[node.args[0]]
        frontier = [node]
        for _ in range(3):
            frontier = [graph.node_table[n] for f in frontier for n in f.parents]
            if any(isinstance(f, RsqrtOp) for f in frontier):
                width = linear.tensor_meta["shape"][-1]
                # One key head (Gemma): its query heads are dealt out a head at
                # a time, and every rank computes the keys and values whole.
                unit = {
                    heads
                    * groups
                    * head_dim: head_dim if heads == 1 else groups * head_dim,
                    heads * head_dim: head_dim,
                }.get(width, 32)
                # The action projection is narrower than a block a rank. The
                # step returns no keys and values: every rank computes those
                # of the one key head whole (a column Linear of one unit).
                if width // unit >= count or (
                    name == "step" and heads == 1 and width == head_dim
                ):
                    columns.add(linear.name)
                    units[linear.name] = unit
                break
    # What every hart computes itself (the masks, the positions and their
    # rotary angles) is computed once on each hart that reads it, not once
    # per segment.
    per_rank = {
        name
        for name in replicated
        if not any(parent in replicated for parent in graph.node_table[name].parents)
    }
    return partition(
        graph,
        params,
        count,
        states,
        replicated,
        columns,
        {},
        per_rank,
        set(),
        set(),
        lambda device, segment: classify(device, segment, name),
        set(),
        units,
        classes,
        True,
    )


def classify(device, segment, name):
    """The graphs of `name` by hart class: on a rank, `attention` (its heads
    of a layer) or `mlp` (its channels); on the reduce hart, `reduce` (the
    partial sums of the ranks, the residuals, the norms and what runs
    whole)."""
    ops = [node for _, node in segment.ops]
    if segment.hart == 0:
        kind = "join" if segment.join else "reduce"
    elif any(isinstance(node, BatchMatmulOp) for node in ops):
        kind = "attention"
    else:
        kind = "mlp"
    return f"{name}_{kind}_{segment.hart_class}"
