from buddy.compiler.graph.operation import MegaKernelOp, OutputOp


def partition(graph):
    operations = list(graph.op_groups["subgraph0"])
    device = graph.group_map_device["subgraph0"]
    groups, current = {}, []
    for operation in operations:
        if isinstance(operation, OutputOp):
            continue
        current.append(operation)
        if isinstance(operation, MegaKernelOp):
            groups[f"subgraph{len(groups)}"] = current
            current = []
    if len(groups) != 9:
        raise ValueError(f"AlexNet expected nine kernels; got {len(groups)}, trailing {[(node.name, type(node).__name__) for node in current]}")
    # The padded final Linear returns its original 1000-class slice.
    groups["subgraph8"].extend(current)
    graph.op_groups = groups
    graph.group_map_device = {name: device for name in groups}
