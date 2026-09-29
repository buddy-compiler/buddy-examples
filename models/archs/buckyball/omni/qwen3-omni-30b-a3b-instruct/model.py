import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .execute import Thinker


class Model(nn.Module):
    # The device owns KV state; this adapter requires no vLLM attention buffers.
    is_attention_free = True
    supports_mrope = True

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.options = vllm_config.additional_config["execution"]
        self.metadata = json.loads(Path(self.options["metadata"]).read_text())
        self.driver = None
        if vllm_config.scheduler_config.max_num_seqs != 1:
            raise ValueError("Thinker device owns one request's KV cache")
        if vllm_config.cache_config.enable_prefix_caching or vllm_config.scheduler_config.enable_chunked_prefill:
            raise ValueError("Thinker device requires complete prefill without prefix reuse")
        if not vllm_config.model_config.enforce_eager or vllm_config.model_config.dtype != torch.float32:
            raise ValueError("BEMU execution requires eager FP32 inputs and outputs")

    def load_device_weights(self):
        if self.driver is not None:
            raise RuntimeError("Thinker weights are already loaded")
        self.driver = Thinker(Path(self.options["directory"]), self.metadata, self.options["timeout"])

    def embed_input_ids(self, input_ids):
        data = self.driver.embeddings(input_ids.detach().cpu().numpy())
        values = np.frombuffer(data, dtype="<f4").copy().reshape(-1, self.metadata["config"]["hidden_size"])
        return torch.from_numpy(values)

    def get_mrope_input_positions(self, input_tokens, mm_features):
        if mm_features:
            raise ValueError("multimodal features require the Media encoder execution path")
        return torch.arange(len(input_tokens), dtype=torch.int64).repeat(3, 1), 0

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if intermediate_tensors is not None or inputs_embeds is not None:
            raise ValueError("Thinker text path requires token IDs")
        tokens = input_ids.detach().cpu().numpy()
        positions = positions.detach().cpu().numpy()
        if positions.ndim == 1:
            positions = np.repeat(positions[None], 3, axis=0)
        start = int(positions[0, 0])
        if not np.array_equal(positions[0], np.arange(start, start + tokens.size)):
            raise ValueError("Thinker text positions must be contiguous")
        result = self.driver.forward(tokens, positions, start)
        if not np.isfinite(result).all():
            raise RuntimeError("non-finite Thinker hidden states")
        return torch.from_numpy(result)

    def compute_logits(self, hidden_states):
        logits = self.driver.logits(hidden_states.detach().cpu().numpy())
        if not np.isfinite(logits).all():
            raise RuntimeError("non-finite Thinker logits")
        return torch.from_numpy(logits)
