"""Offline native INT5 weight policy for the Poly Qwen3-8B recipe."""

import torch


def quantize_weight(values, axis):
    x = values.float().movedim(axis, -1).contiguous()
    if x.shape[-1] % 32 or not torch.all(torch.isfinite(x)):
        raise ValueError("INT5 weights require finite complete 32-value blocks")
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    maximum = blocks.abs().amax(-1, keepdim=True)
    exponent = (
        maximum.clamp_min(torch.finfo(torch.float32).tiny).log2().floor() - 6
    ).clamp(-127, 127)
    exponent = torch.where(maximum == 0, 0, exponent)
    base = torch.exp2(exponent)
    factor = torch.round(maximum / base * 16 / 15).clamp_min(1)
    codes = torch.round(blocks / base * 16 / factor).clamp(-15, 15).long()
    groups = (codes & 31).reshape(*codes.shape[:-1], 4, 8)
    words = sum(groups[..., i] << (5 * i) for i in range(8))
    packed = torch.stack([((words >> (8 * i)) & 255) for i in range(5)], -1)
    packed = packed.flatten(-3).to(torch.uint8)
    factors = factor[..., 0].to(torch.uint8)
    exponents = (exponent[..., 0] + 127).to(torch.uint8)
    return (
        torch.cat([packed, factors, exponents], -1).view(torch.int8).movedim(-1, axis)
    )


def quantize(graph, params):
    from buddy.compiler.graph.operation import OutputOp, TransposeMatmulFusedOp
    from compiler.frontend.quant import MXINT5ToFP32DequantOp
    from frontend.quant.graph import quantize as quantize_mxint8

    originals = dict(zip((node.name for node in graph.params), params, strict=True))
    params = quantize_mxint8(graph, params)
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    (output,) = [node for node in graph.body if isinstance(node, OutputOp)]
    head = graph.node_table[graph.node_table[output.args[0]].args[0]]
    if not isinstance(head, TransposeMatmulFusedOp):
        raise ValueError("Expected the decoder's final Linear behind its logits view")
    for op in list(graph.body):
        if not isinstance(op, TransposeMatmulFusedOp) or op is head:
            continue
        restored = graph.node_table[op.args[1]]
        weight = graph.node_table[restored.args[0]]
        packed = quantize_weight(originals[weight.name], 1)
        values[weight.name] = packed
        weight.tensor_meta["shape"] = list(packed.shape)
        decoded = MXINT5ToFP32DequantOp()
        decoded.name = restored.name
        graph.displace_node(restored, decoded)
    return [values[node.name] for node in graph.params]
