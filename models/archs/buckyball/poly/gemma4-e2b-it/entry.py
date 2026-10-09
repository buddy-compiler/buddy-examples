import torch
from torch import nn

from stack.compiler.quant import mxfp8_cache as codec


@torch.library.custom_op("gemma::row_embedding", mutates_args=())
def row_embedding(
    weight: torch.Tensor, indices: torch.Tensor, width: int
) -> torch.Tensor:
    rows = weight[indices]
    return codec.decode(rows[..., :width], rows[..., width:])


@row_embedding.register_fake
def _row_embedding_fake(weight, indices, width):
    if (
        weight.dtype != torch.int8
        or weight.ndim != 2
        or width <= 0
        or width % 32
        or weight.shape[1] != width + width // 32
        or indices.dtype != torch.int64
        or indices.ndim != 2
    ):
        raise ValueError(
            "Gemma entry requires row-packed I8 embeddings and I64 token IDs"
        )
    return weight.new_empty((*indices.shape, width), dtype=torch.float32)


class Embedding(nn.Module):
    def __init__(self, embedding, packed):
        super().__init__()
        self.register_buffer("weight", packed)
        self.register_buffer("scale", embedding.embed_scale)
        self.width = embedding.embedding_dim

    def forward(self, tokens):
        return row_embedding(self.weight, tokens, self.width) * self.scale


class PerLayerInputs(nn.Module):
    def __init__(self, model, projection):
        super().__init__()
        self.projection = projection
        self.norm = model.per_layer_projection_norm
        self.layers = model.config.num_hidden_layers
        self.width = model.hidden_size_per_layer_input
        self.projection_scale = model.per_layer_model_projection_scale
        self.input_scale = model.per_layer_input_scale

    def forward(self, hidden, embedding):
        projected = self.projection(hidden) * self.projection_scale
        shape = (*hidden.shape[:-1], self.layers, self.width)
        return (
            (self.norm(projected.reshape(shape)) + embedding.reshape(shape))
            * self.input_scale
        ).flatten(2)


class Positions(nn.Module):
    def __init__(self, model, kind):
        super().__init__()
        self.rotary = model.rotary_emb
        self.kind = kind

    def forward(self, hidden, positions):
        cosine, sine = self.rotary(hidden, positions, self.kind)
        keys = torch.arange(512, device=positions.device)[None, None, None, :]
        visible = keys <= positions[:, None, :, None]
        if self.kind == "sliding_attention":
            visible = visible & (keys > positions[:, None, :, None] - 512)
        return cosine, sine, torch.where(visible, 0.0, torch.finfo(torch.float32).min)


class FinalNorm(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.norm = model.norm

    def forward(self, hidden):
        return self.norm(hidden)


class Logits(nn.Module):
    def __init__(self, model, projection):
        super().__init__()
        self.projection = projection
        self.softcap = model.config.final_logit_softcapping

    def forward(self, hidden):
        logits = self.projection(hidden)
        return torch.tanh(logits / self.softcap) * self.softcap


def register(compiler):
    from buddy.compiler.graph.operation import Op
    from stack.compiler.quant.mxfp8_embedding import MXFP8EmbeddingOp, lower_embedding

    class RowEmbeddingOp(Op):
        pass

    def lower(node, symbols):
        embedding = MXFP8EmbeddingOp()
        embedding._arguments = node.args[:2]
        embedding._tensor_meta = dict(node.tensor_meta)
        return lower_embedding(embedding, symbols)

    compiler._ops_map["row_embedding.default"] = RowEmbeddingOp
    compiler._ops_registry["RowEmbeddingOp"] = lower
