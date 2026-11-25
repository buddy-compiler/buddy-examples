"""pi0-FAST as the three classes of graphs piplus runs: the vision encoder
(once per camera), the prefill (once: the images and the prompt to the keys
and values of every layer and the first token) and one decoding step (once
per token: the token before it to the next)."""

import copy
import math

import torch
from lerobot.utils.constants import OPENPI_ATTENTION_MASK_VALUE
from transformers import DynamicCache
from transformers.models.gemma import modeling_gemma

# The tokens a decoding run generates at most (the policy's
# max_decoding_steps), and the keys a decoding step brings itself: a block
# of 32, the first its token's, the rest never seen.
generated = 256
own = 32

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
    gemma = flow.paligemma_with_expert.paligemma.model.language_model.config
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


def rows(flow, tokens):
    """The rows of the embedding table of `tokens`, as the runner reads them
    for the graphs: the table is in the model image once, in no graph."""
    table = (
        flow.paligemma_with_expert.paligemma.model.language_model.get_input_embeddings()
    )
    return table.weight[tokens]


def scaled(flow):
    """The embedding of the rows of the table the runner read (Gemma's
    embedding without its table)."""
    table = (
        flow.paligemma_with_expert.paligemma.model.language_model.get_input_embeddings()
    )
    scale = float(table.embed_scale)
    return lambda rows: rows * scale


class Prefill(torch.nn.Module):
    """The rows of the embedding table of the prompt and of the token the
    model continues (model.Start), their mask and the image tokens of every
    camera to the logits of the first token,
    the position of the next and the bias of the scores of the decoding
    steps (the prefix's keys as its padding leaves them, no generated key
    seen yet, of a step's own block its first), and the keys and values of
    every layer, which the decoding steps read. The tokens of each camera
    are an input of their own: the cameras are encoded apart, on any hart."""

    def __init__(self, flow):
        super().__init__()
        # The prefill embeds the image tokens Vision computed: a copy of the
        # flow whose embed_image passes them through. The copies get their
        # own submodule tables, which copy.copy would share with the flow.
        self.flow = copy.copy(flow)
        self.flow._modules = dict(flow._modules)
        self.flow.paligemma_with_expert = copy.copy(flow.paligemma_with_expert)
        self.flow.paligemma_with_expert._modules = dict(
            flow.paligemma_with_expert._modules
        )
        self.flow.paligemma_with_expert.embed_image = lambda tokens: tokens
        self.flow.paligemma_with_expert.embed_language_tokens = scaled(flow)

    def forward(self, rows, masks, *images):
        present = [torch.ones(1, dtype=torch.bool)] * len(images)
        embeddings, pad_masks, seen, _, _ = self.flow.embed_prefix_fast(
            list(images),
            present,
            rows,
            masks.bool(),
            fast_action_tokens=None,
            fast_action_masks=None,
        )
        last = embeddings.shape[1] - 1
        # The prefix gains masked tokens up to a multiple of 32: its
        # attention takes no padding of its scores and values in any layer.
        tail = -embeddings.shape[1] % 32
        embeddings = torch.nn.functional.pad(embeddings, (0, 0, 0, tail))
        pad_masks = torch.nn.functional.pad(pad_masks, (0, tail), value=False)
        seen = torch.nn.functional.pad(seen, (0, tail, 0, tail), value=False)
        (hidden, _), cache = self.flow.paligemma_with_expert.forward(
            attention_mask=biased(self.flow, seen),
            position_ids=torch.cumsum(pad_masks, dim=1) - 1,
            past_key_values=None,
            inputs_embeds=[embeddings, None],
            use_cache=True,
        )
        logits = self.flow.paligemma_with_expert.paligemma.lm_head(
            hidden[:, last : last + 1]
        )
        block = torch.zeros(1, own, dtype=torch.bool)
        block[:, 0] = True
        keys = torch.cat(
            [pad_masks, torch.zeros(1, generated, dtype=torch.bool), block], dim=1
        )
        return (
            logits,
            pad_masks.sum(-1, keepdim=True),
            biased(self.flow, keys[:, None]),
            *[layer.keys for layer in cache.layers],
            *[layer.values for layer in cache.layers],
        )


class DecodeCache(DynamicCache):
    """The keys and values a decoding step attends to, one after the other
    along the keys: the prefix's, those of the tokens generated before it
    (the rows the runner filled) and its own, the first of a block."""

    def __init__(self, prefix, before):
        super().__init__()
        self.prefix = prefix
        self.before = before
        self.own = []

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        self.own.append((key_states, value_states))
        prefix_key, prefix_value = self.prefix[layer_idx]
        before_key, before_value = self.before[layer_idx]
        rest = (0, 0, 0, own - 1)
        return (
            (prefix_key, before_key, torch.nn.functional.pad(key_states, rest)),
            (prefix_value, before_value, torch.nn.functional.pad(value_states, rest)),
        )

    def get_seq_length(self, layer_idx=0):
        return self.prefix[layer_idx][0].shape[2] + self.before[layer_idx][0].shape[2]


class Decode(torch.nn.Module):
    """One decoding step: the row of the embedding table of the token before,
    its position, the bias of its scores, the prefix's keys and values and
    those of the tokens generated before it to the logits of the next token
    and its own keys and values, which the runner writes after the generated
    ones."""

    def __init__(self, flow):
        super().__init__()
        # Its own language model: the weights of a step are single MXINT8
        # terms (quant.graph.doubled takes the policy's), half the weights
        # a token reads.
        self.flow = copy.deepcopy(flow)
        self.scaled = scaled(flow)

    def forward(self, row, position, bias, *cache):
        layers = len(cache) // 4
        past = DecodeCache(
            [(cache[layer], cache[layers + layer]) for layer in range(layers)],
            [
                (cache[2 * layers + layer], cache[3 * layers + layer])
                for layer in range(layers)
            ],
        )
        (hidden, _), _ = self.flow.paligemma_with_expert.forward(
            attention_mask=bias,
            position_ids=position,
            past_key_values=past,
            inputs_embeds=[self.scaled(row), None],
            use_cache=True,
        )
        return (
            self.flow.paligemma_with_expert.paligemma.lm_head(hidden),
            *[key for key, _ in past.own],
            *[value for _, value in past.own],
        )
