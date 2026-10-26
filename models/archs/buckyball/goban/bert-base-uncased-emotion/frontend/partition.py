def partition(graph, count):
    """Split an ordered group without separating a tuple producer from its users."""
    operations = list(graph.op_groups["subgraph0"])
    positions = {node.name: index for index, node in enumerate(graph.body)}
    operations.sort(key=lambda node: positions[node.name])
    positions = {node.name: index for index, node in enumerate(operations)}
    if not 1 <= count <= len(operations):
        raise ValueError("Partition count must fit the operation count")
    cuts = set(range(1, len(operations) + 1))
    for index, node in enumerate(operations):
        if isinstance(node.tensor_meta.get("dtype"), (tuple, list)):
            end = max(
                [index]
                + [positions[name] for name in node._children if name in positions]
            )
            cuts.difference_update(range(index + 1, end + 1))
    base, remainder = divmod(len(operations), count)
    groups, offset = {}, 0
    for index in range(count):
        end = (
            len(operations)
            if index + 1 == count
            else offset + base + (index < remainder)
        )
        while end < len(operations) and end not in cuts:
            end += 1
        if end > len(operations) - (count - index - 1):
            raise ValueError("Cannot partition without splitting a tuple result")
        groups[f"subgraph{index}"] = operations[offset:end]
        offset = end
    device = graph.group_map_device["subgraph0"]
    graph.op_groups = groups
    graph.group_map_device = dict.fromkeys(groups, device)
