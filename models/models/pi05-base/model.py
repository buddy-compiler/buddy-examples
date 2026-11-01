# ===- model.py ----------------------------------------------------------------
#
# pi0.5 as one graph: PaliGemma runs once over the images and the prompt (which
# carries the discretized robot state); the action expert then integrates the
# flow from the noise to an action chunk in ten Euler steps.
#
# ===---------------------------------------------------------------------------

from pathlib import Path

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05 import modeling_pi05
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from transformers import DynamicCache
from transformers.models.gemma import modeling_gemma

checkpoint = Path(__file__).resolve().parent / "output/pi05-base"
config = PreTrainedConfig.from_pretrained(checkpoint)
config.device = "cpu"
gemma_attention = modeling_gemma.eager_attention_forward


class PrefixCache(DynamicCache):
    """The keys and values of the prefix, read by the action expert in every
    step: the keys and values of the actions are not appended, so no step
    copies the prefix."""

    def __init__(self, cache):
        super().__init__()
        self.prefix = [(layer.keys, layer.values) for layer in cache.layers]

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        prefix_key, prefix_value = self.prefix[layer_idx]
        return (prefix_key, key_states), (prefix_value, value_states)

    def get_seq_length(self, layer_idx=0):
        return self.prefix[layer_idx][0].shape[2]


def attention(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
):
    """The action queries attend to the prefix and to the actions as two
    blocks of one softmax; the query heads of the key/value head attend to
    it together."""
    if not isinstance(key, tuple):
        return gemma_attention(
            module, query, key, value, attention_mask, scaling, dropout, **kwargs
        )
    (prefix_key, key), (prefix_value, value) = key, value
    batch, heads, length, dim = query.shape
    kv_heads = key.shape[1]
    groups = heads // kv_heads
    prefix = prefix_key.shape[2]
    query = query.reshape(batch, kv_heads, groups * length, dim)
    prefix_scores = torch.matmul(query, prefix_key.transpose(2, 3)) * scaling
    prefix_scores = prefix_scores.view(batch, kv_heads, groups, length, prefix)
    prefix_scores = prefix_scores + attention_mask[:, :, None, :, :prefix]
    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    scores = scores.view(batch, kv_heads, groups, length, key.shape[2])
    scores = scores + attention_mask[:, :, None, :, prefix:]
    peak = torch.maximum(
        prefix_scores.amax(-1, keepdim=True), scores.amax(-1, keepdim=True)
    )
    prefix_scores = torch.exp(prefix_scores - peak)
    scores = torch.exp(scores - peak)
    total = prefix_scores.sum(-1, keepdim=True) + scores.sum(-1, keepdim=True)
    output = torch.matmul(
        prefix_scores.view(batch, kv_heads, groups * length, prefix), prefix_value
    )
    output = output + torch.matmul(
        scores.view(batch, kv_heads, groups * length, key.shape[2]), value
    )
    output = output.view(batch, kv_heads, groups, length, dim) / total
    return output.view(batch, heads, length, dim).transpose(1, 2).contiguous(), None


class Policy(torch.nn.Module):
    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def forward(self, images, tokens, masks, noise):
        img_masks = [torch.ones(1, dtype=torch.bool)] * images.shape[0]
        return self.flow.sample_actions(
            list(images.unbind(0)), img_masks, tokens, masks.bool(), noise=noise
        )


def load():
    """The policy in FP32, whatever the checkpoint stores, with the rewrites
    above in place."""
    policy = PI05Policy.from_pretrained(checkpoint, config=config)
    flow = policy.float().eval().model
    # Both Gemmas attend through the eager implementation the rewrite replaces;
    # the policy switches to it only inside its own calls.
    language_model = flow.paligemma_with_expert.paligemma.model.language_model
    language_model.config._attn_implementation = "eager"
    flow.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
    modeling_pi05.clone_past_key_values = PrefixCache
    modeling_gemma.eager_attention_forward = attention
    return Policy(flow).eval()


def inputs():
    """Example inputs: the camera images in [-1, 1], the prompt tokens (with
    the discretized state) and their mask, and the noise of the action chunk."""
    cameras = len(config.image_features)
    height, width = config.image_resolution
    return (
        torch.zeros((cameras, 1, 3, height, width)),
        torch.zeros((1, config.tokenizer_max_length), dtype=torch.int64),
        torch.ones((1, config.tokenizer_max_length), dtype=torch.int64),
        torch.zeros((1, config.chunk_size, config.max_action_dim)),
    )
