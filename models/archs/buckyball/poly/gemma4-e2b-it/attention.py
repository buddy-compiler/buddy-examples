import torch
from torch import nn
from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb

from stack.compiler.quant import mxfp8_cache as codec


@torch.library.custom_op("gemma::prefill_head_attention", mutates_args=())
def prefill_head_attention(
    query: torch.Tensor,
    key_codes: torch.Tensor,
    key_scales: torch.Tensor,
    value_codes: torch.Tensor,
    value_scales: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    key = codec.decode(key_codes, key_scales)
    value = codec.decode(value_codes, value_scales)
    repeats = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(repeats, dim=1)
    value = value.repeat_interleave(repeats, dim=1)
    scores = torch.matmul(query, key.transpose(-1, -2)) * scale
    probabilities = nn.functional.softmax(scores + mask, dim=-1, dtype=torch.float32)
    return torch.matmul(probabilities, value).transpose(1, 2).flatten(2)


@prefill_head_attention.register_fake
def _prefill_head_attention_fake(
    query, key_codes, key_scales, value_codes, value_scales, mask, scale
):
    if (
        query.ndim != 4
        or key_codes.ndim != 4
        or key_codes.shape != value_codes.shape
        or key_scales.shape != value_scales.shape
        or key_scales.shape != (*key_codes.shape[:-1], key_codes.shape[-1] // 32)
        or query.shape[0] != key_codes.shape[0]
        or query.shape[1] % key_codes.shape[1]
        or query.shape[-1] != key_codes.shape[-1]
        or query.shape[-1] % 32
        or query.dtype != torch.float32
        or any(
            t.dtype != torch.int8
            for t in (key_codes, key_scales, value_codes, value_scales)
        )
        or mask.shape != (query.shape[0], 1, query.shape[2], key_codes.shape[2])
        or mask.dtype != torch.float32
    ):
        raise ValueError("Gemma prefill attention has invalid compressed KV shapes")
    return query.new_empty(
        (query.shape[0], query.shape[2], query.shape[1] * query.shape[3])
    )


@torch.library.custom_op("gemma::decode_probabilities", mutates_args=())
def decode_probabilities(scores: torch.Tensor) -> torch.Tensor:
    maximum = scores.amax(dim=-1, keepdim=True)
    total = torch.exp(scores - maximum).sum(dim=-1, keepdim=True)
    return torch.exp(scores - (maximum + torch.log(total)))


@decode_probabilities.register_fake
def _decode_probabilities_fake(scores):
    return scores.new_empty(scores.shape)


class AttentionHeads(nn.Module):
    def __init__(self, layer, first_head, end_head, *, prefill):
        super().__init__()
        if not 0 <= first_head < end_head <= layer.self_attn.config.num_attention_heads:
            raise ValueError("Empty Gemma head ranges have no compute task")
        self.norm = layer.self_attn.q_norm
        self.width = layer.self_attn.head_dim
        self.first, self.end = first_head, end_head
        self.scale = layer.self_attn.scaling
        self.prefill = prefill

    def forward(
        self,
        query,
        cosine,
        sine,
        key_codes,
        key_scales,
        value_codes,
        value_scales,
        mask,
    ):
        shape = (*query.shape[:-1], -1, self.width)
        query = query.reshape(shape)[:, :, self.first : self.end, :]
        query = self.norm(query)
        query = apply_rotary_pos_emb(query, cosine, sine, unsqueeze_dim=2).transpose(
            1, 2
        )
        if self.prefill:
            return prefill_head_attention(
                query,
                key_codes,
                key_scales,
                value_codes,
                value_scales,
                mask,
                self.scale,
            )
        key, value = codec.decode(key_codes, key_scales), codec.decode(
            value_codes, value_scales
        )
        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        scores = scores + mask
        probabilities = decode_probabilities(scores)
        return torch.matmul(probabilities, value).transpose(1, 2).flatten(2)


def register(compiler):
    from buddy.compiler.graph.operation import Op
    from buddy_mlir import ir
    from buddy_mlir.dialects import bufferization, func, memref
    from .attention_mxfp8 import lower_prefill_attention

    class PrefillHeadAttentionOp(Op):
        pass

    class DecodeProbabilitiesOp(Op):
        pass

    def lower_probabilities(node, symbols):
        value = symbols[(node.args[0], 0)]
        tensor = ir.RankedTensorType(value.type)
        f32 = ir.F32Type.get()
        if tensor.element_type != f32 or tensor.rank != 4:
            raise ValueError("Gemma decode probabilities require an FP32 score tensor")
        buffer = bufferization.ToBufferOp(
            ir.MemRefType.get(tensor.shape, f32), value, read_only=True
        ).result
        output = memref.AllocOp(ir.MemRefType.get(tensor.shape, f32), [], []).result
        unranked = ir.UnrankedMemRefType.get(f32, None)
        arguments = [memref.CastOp(unranked, arg).result for arg in (output, buffer)]
        module = ir.InsertionPoint.current.block.owner
        while module.operation.name != "builtin.module":
            module = module.operation.parent
        table = ir.SymbolTable(module)
        name = "rvv_logsumexp_softmax"
        function_type = ir.FunctionType.get([arg.type for arg in arguments], [])
        if name not in table:
            with ir.InsertionPoint.at_block_begin(module.regions[0].blocks[0]):
                function = func.FuncOp(name, function_type, visibility="private")
                function.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
        elif (
            ir.TypeAttr(table[name].attributes["function_type"]).value != function_type
        ):
            raise ValueError("Gemma decode probabilities callee type mismatch")
        func.CallOp([], name, arguments)
        return bufferization.ToTensorOp(
            tensor, output, restrict=True, writable=True
        ).result

    compiler._ops_map["prefill_head_attention.default"] = PrefillHeadAttentionOp
    compiler._ops_registry["PrefillHeadAttentionOp"] = lower_prefill_attention
    compiler._ops_map["decode_probabilities.default"] = DecodeProbabilitiesOp
    compiler._ops_registry["DecodeProbabilitiesOp"] = lower_probabilities
