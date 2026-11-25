"""pi0 as the three classes of graphs piplus runs: the vision encoder (once
per camera), the prefix (once) and one denoising step (once per step)."""

import copy
import math
from types import SimpleNamespace

import torch
from lerobot.policies.pi0.modeling_pi0 import (
    OPENPI_ATTENTION_MASK_VALUE,
    make_att_2d_masks,
)
from transformers.models.gemma import modeling_gemma

# The scores get their bias before their scale: softmax((q . k^T + bias) *
# scaling), the bias the mask of the policy over the scaling (a power of two,
# so the sum is the policy's) and no lower than -2^100, which a product can
# be added to. The thread of piplus that computes an attention starts the
# scores from the bias.
lowest = -(2.0**100)


def attention(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
):
    """The attention of both Gemmas: the query heads of a key head attend to
    its keys together, as rows of one product (a rank of the device takes
    some of the query heads, every rank the keys and values). A step attends
    to the prefix's keys and to its own (PrefixCache), one after the other
    along the keys: the device reads each part where it is."""
    if isinstance(key, tuple):
        key, value = torch.cat(key, dim=2), torch.cat(value, dim=2)
    batch, heads, length, dim = query.shape
    kv_heads, total = key.shape[1], key.shape[2]
    groups = heads // kv_heads
    query = query.reshape(batch, kv_heads, groups * length, dim)
    scores = torch.matmul(query, key.transpose(2, 3))
    scores = scores.view(batch, kv_heads, groups, length, total)
    weights = torch.softmax((scores + attention_mask[:, :, None]) * scaling, dim=-1)
    weights = weights.view(batch, kv_heads, groups * length, total)
    output = torch.matmul(weights, value)
    return output.view(batch, heads, length, dim).transpose(1, 2).contiguous(), None


def angles(module, x, position_ids):
    """The rotary angles of both Gemmas (GemmaRotaryEmbedding.forward): the
    product of every position by every frequency element by element, not a
    matrix product, which the device would quantize."""
    frequencies = (
        position_ids[:, :, None].float() * module.inv_freq[None, None, :].float()
    )
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cos = embedding.cos() * module.attention_scaling
    sin = embedding.sin() * module.attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# The turned queries and keys of the attention being computed, in the order
# its rotary embedding reads them.
turned = []


class Turned(torch.nn.Module):
    """A query or key projection of a Gemma and the same projection with the
    halves of every head turned (rotate_half of its output: the rows of a
    head's second half negated, then those of its first), which the rotary
    embedding reads beside it: no hart joins halves of a row."""

    def __init__(self, projection, head_dim):
        super().__init__()
        self.projection = projection
        weight = projection.weight.view(-1, 2, head_dim // 2, projection.in_features)
        self.rotated = torch.nn.Linear(
            projection.in_features, projection.out_features, bias=False
        )
        self.rotated.weight = torch.nn.Parameter(
            torch.cat((-weight[:, 1:], weight[:, :1]), dim=1).reshape(
                projection.weight.shape
            )
        )

    @property
    def weight(self):
        return self.projection.weight

    def forward(self, x):
        turned.append(self.rotated(x))
        return self.projection(x)


def rotate_half(x):
    """rotate_half of the queries or keys x [batch, heads, length, head_dim]:
    their turned projection."""
    batch, heads, length, dim = x.shape
    return turned.pop(0).view(batch, length, heads, dim).transpose(1, 2)


def rewrite(*gemmas):
    """The Gemmas attend and encode their positions as the device computes
    them."""
    modeling_gemma.eager_attention_forward = attention
    modeling_gemma.GemmaRotaryEmbedding.forward = angles
    modeling_gemma.rotate_half = rotate_half
    for gemma in gemmas:
        for layer in gemma.layers:
            attention_ = layer.self_attn
            attention_.q_proj = Turned(attention_.q_proj, attention_.head_dim)
            attention_.k_proj = Turned(attention_.k_proj, attention_.head_dim)


def biased(flow, seen):
    """The bias of the scores of the keys `seen` [1, queries, keys]."""
    gemma = flow.paligemma_with_expert.gemma_expert.model.config
    bias = torch.where(seen[:, None], 0.0, OPENPI_ATTENTION_MASK_VALUE)
    return (bias * gemma.head_dim**0.5).clamp(min=lowest)


def padded(linear, heads, dim, factor=1.0):
    """A projection to the heads of an attention with every head padded with
    zeros to `dim` values, its values times `factor`."""
    width = linear.out_features // heads
    wide = torch.nn.Linear(linear.in_features, heads * dim)
    weight = torch.zeros(heads, dim, linear.in_features)
    weight[:, :width] = linear.weight.view(heads, width, -1) * factor
    bias = torch.zeros(heads, dim)
    bias[:, :width] = linear.bias.view(heads, width) * factor
    wide.weight = torch.nn.Parameter(weight.reshape(heads * dim, -1))
    wide.bias = torch.nn.Parameter(bias.reshape(-1))
    return wide


def widen(attention):
    """A SigLIP attention as the attention of the device takes it: its heads
    (72 values) padded with zeros to whole blocks of 32 values, in the
    weights of the projections (the padded keys and values are zero: the
    scores and the outputs are those of the heads, the output projection
    reads the padding with zero weights), and the scores scaled by a power
    of two, the rest of the scale in the queries."""
    heads, width = attention.num_heads, attention.head_dim
    dim = -(-width // 32) * 32
    scale = 2.0 ** round(math.log2(attention.scale))
    out = attention.out_proj
    wide = torch.nn.Linear(heads * dim, out.out_features)
    weight = torch.zeros(out.out_features, heads, dim)
    weight[:, :, :width] = out.weight.view(-1, heads, width)
    wide.weight = torch.nn.Parameter(weight.reshape(out.out_features, -1))
    wide.bias = out.bias
    attention.q_proj = padded(attention.q_proj, heads, dim, attention.scale / scale)
    attention.k_proj = padded(attention.k_proj, heads, dim)
    attention.v_proj = padded(attention.v_proj, heads, dim)
    attention.out_proj = wide
    attention.head_dim = dim
    attention.scale = scale


class Vision(torch.nn.Module):
    """One camera image to its image tokens: SigLIP and the projector into
    the language model."""

    def __init__(self, flow):
        super().__init__()
        self.flow = flow
        # Its own SigLIP, the heads widened (widen): the policy keeps its.
        model = flow.paligemma_with_expert.paligemma.model
        self.vision = copy.deepcopy(model.vision_tower.vision_model)
        for layer in self.vision.encoder.layers:
            widen(layer.self_attn)

    def forward(self, image):
        # embed_image, with the embeddings as a projection of the patches in
        # token order: the patch convolution in channel order, flattened and
        # transposed back (SiglipVisionEmbeddings), is a transpose the hart
        # CPU makes element by element. The position ids are the patch
        # indices, so the whole position table is added.
        model = self.flow.paligemma_with_expert.paligemma.model
        vision = self.vision
        embeddings = vision.embeddings
        size = embeddings.patch_size
        batch, channels, height, width = image.shape
        patches = image.reshape(
            batch, channels, height // size, size, width // size, size
        )
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(
            batch, (height // size) * (width // size), channels * size * size
        )
        convolution = embeddings.patch_embedding
        hidden = torch.nn.functional.linear(
            patches,
            convolution.weight.reshape(convolution.out_channels, -1),
            convolution.bias,
        )
        hidden = hidden + embeddings.position_embedding.weight
        hidden = vision.encoder(inputs_embeds=hidden).last_hidden_state
        return model.multi_modal_projector(vision.post_layernorm(hidden))


class Prefix(torch.nn.Module):
    """The prompt and the image tokens of every camera to the keys and values
    of every language model layer, which the denoising steps read, with the
    bias of the scores and the positions of every step. The tokens of each
    camera are an input of their own: the cameras are encoded apart, on any
    hart."""

    def __init__(self, flow):
        super().__init__()
        # The prefix embeds the image tokens Vision computed: a copy of the
        # flow whose embed_image passes them through. The copies get their
        # own submodule tables, which copy.copy would share with the flow.
        self.flow = copy.copy(flow)
        self.flow._modules = dict(flow._modules)
        self.flow.paligemma_with_expert = copy.copy(flow.paligemma_with_expert)
        self.flow.paligemma_with_expert._modules = dict(
            flow.paligemma_with_expert._modules
        )
        self.flow.paligemma_with_expert.embed_image = lambda tokens: tokens
        # A step's tokens: the state, then the actions, padded to a multiple
        # of 32.
        self.suffix = 1 + flow.config.chunk_size
        self.chunk = -(-self.suffix // 32) * 32

    def forward(self, tokens, masks, *images):
        present = [torch.ones(1, dtype=torch.bool)] * len(images)
        embeddings, pad_masks, att_masks = self.flow.embed_prefix(
            list(images), present, tokens, masks.bool()
        )
        # The prefix gains masked tokens up to a multiple of 32: its
        # attention takes no padding of its scores and values in any layer.
        tail = -embeddings.shape[1] % 32
        embeddings = torch.nn.functional.pad(embeddings, (0, 0, 0, tail))
        pad_masks = torch.nn.functional.pad(pad_masks, (0, tail), value=False)
        att_masks = torch.nn.functional.pad(att_masks, (0, tail))
        _, cache = self.flow.paligemma_with_expert.forward(
            attention_mask=biased(self.flow, make_att_2d_masks(pad_masks, att_masks)),
            position_ids=torch.cumsum(pad_masks, dim=1) - 1,
            past_key_values=None,
            inputs_embeds=[embeddings, None],
            use_cache=True,
        )
        # What every denoising step attends with, once: the bias of its
        # scores (the prefix's tokens, then its chunk, whose padding no
        # action sees) and the positions of its chunk, which depend on the
        # prefix only.
        suffix = torch.zeros(1, self.chunk, dtype=torch.bool)
        suffix[:, : self.suffix] = True
        # The state starts a block and the actions another.
        first = torch.zeros(1, self.chunk)
        first[:, :2] = 1
        seen = torch.cat(
            [
                pad_masks[:, None, :].expand(1, self.chunk, pad_masks.shape[1]),
                make_att_2d_masks(suffix, first),
            ],
            dim=2,
        )
        bias = biased(self.flow, seen)
        positions = pad_masks.sum(-1)[:, None] + torch.cumsum(suffix, dim=1) - 1
        # The rotary cos and sin of those positions [2, 1, chunk, head_dim]:
        # no step computes them again.
        expert = self.flow.paligemma_with_expert.gemma_expert.model
        positions = torch.stack(angles(expert.rotary_emb, bias, positions))
        return (
            bias,
            positions,
            *[layer.keys for layer in cache.layers],
            *[layer.values for layer in cache.layers],
        )


class Rotated(torch.nn.Module):
    """The rotary embedding of a step: the cos and sin the prefix computed
    for its positions."""

    def forward(self, x, positions):
        return positions[0], positions[1]


class Step(torch.nn.Module):
    """One denoising step: the actions x_t at time t, the state, the prefix's
    bias and positions for the steps and its keys and values to the velocity
    of the actions."""

    def __init__(self, flow, cache):
        super().__init__()
        # The state and the chunk of actions padded to a multiple of 32
        # tokens: with the prefix's, the keys of the attention (the prefix,
        # the state and the chunk) need no padding. No action sees the
        # padding tokens (Prefix), so they change no action; their
        # velocities are dropped.
        self.flow = copy.deepcopy(flow)
        self.actions = flow.config.chunk_size
        self.flow.config.chunk_size = -(-(1 + self.actions) // 32) * 32 - 1
        self.cache = cache
        self.flow.paligemma_with_expert.gemma_expert.model.rotary_emb = Rotated()

    def forward(self, x_t, time, state, bias, positions, *cache):
        half = len(cache) // 2
        # The prefix's keys and values as they are: DynamicCache.update would
        # concatenate each to an empty tensor, a copy of the whole cache.
        past = SimpleNamespace(
            layers=[
                SimpleNamespace(keys=cache[layer], values=cache[half + layer])
                for layer in range(half)
            ]
        )
        x_t = torch.nn.functional.pad(
            x_t, (0, 0, 0, self.flow.config.chunk_size - self.actions)
        )
        suffix, _, _, _ = self.flow.embed_suffix(state, x_t, time)
        outputs, _ = self.flow.paligemma_with_expert.forward(
            attention_mask=bias,
            position_ids=positions,
            past_key_values=self.cache(past),
            inputs_embeds=[None, suffix],
            use_cache=False,
            adarms_cond=[None, None],
        )
        velocity = self.flow.action_out_proj(
            outputs[1][:, -self.flow.config.chunk_size :].to(dtype=torch.float32)
        )
        return velocity[:, : self.actions]
