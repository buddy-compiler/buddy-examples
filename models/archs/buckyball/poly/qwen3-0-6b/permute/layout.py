from examples.balls.mxmm.compiler.python import grouped_attention


def apply(graph, *, kind):
    if kind == "attention":
        grouped_attention.apply(graph)
