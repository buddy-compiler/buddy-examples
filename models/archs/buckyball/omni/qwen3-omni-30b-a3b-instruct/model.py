import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerMultiModalProcessor,
    Qwen3OmniMoeThinkerProcessingInfo,
    Qwen3OmniMoeThinkerDummyInputsBuilder,
)

from .execute import Thinker
from .media.embeddings import Encoders
from .media.positions import positions as multimodal_positions


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class Model(nn.Module, SupportsMultiModal):
    # The device owns KV state; this adapter requires no vLLM attention buffers.
    supports_multimodal_raw_input_only = True
    is_attention_free = True
    supports_mrope = True

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.options = vllm_config.additional_config["execution"]
        self.metadata = json.loads(Path(self.options["metadata"]).read_text())
        self.driver = None
        self.media = None
        self.deepstack = None
        self.next_position = 0
        self.config = vllm_config.model_config.hf_config.thinker_config
        if vllm_config.scheduler_config.max_num_seqs != 1:
            raise ValueError("Thinker device owns one request's KV cache")
        if (
            vllm_config.cache_config.enable_prefix_caching
            or vllm_config.scheduler_config.enable_chunked_prefill
        ):
            raise ValueError(
                "Thinker device requires complete prefill without prefix reuse"
            )
        if (
            not vllm_config.model_config.enforce_eager
            or vllm_config.model_config.dtype != torch.float32
        ):
            raise ValueError("BEMU execution requires eager FP32 inputs and outputs")

    def load_device_weights(self):
        if self.driver is not None:
            raise RuntimeError("Thinker weights are already loaded")
        directory = Path(self.options["directory"])
        with ThreadPoolExecutor(max_workers=2) as pool:
            thinker = pool.submit(
                Thinker, directory, self.metadata, self.options["timeout"]
            )
            encoders = pool.submit(
                Encoders, directory, self.metadata, self.options["timeout"]
            )
            self.driver = thinker.result()
            self.media = encoders.result()

    def get_language_model(self):
        return self

    @classmethod
    def get_placeholder_str(cls, modality, i):
        if modality in ("image", "video"):
            return f"<|vision_start|><|{modality}_pad|><|vision_end|>"
        if modality == "audio":
            return "<|audio_start|><|audio_pad|><|audio_end|>"
        raise ValueError(f"unknown modality: {modality}")

    def embed_multimodal(self, **kwargs):
        return self.media.encode(**kwargs)

    def embed_input_ids(
        self, input_ids, multimodal_embeddings=None, *, is_multimodal=None
    ):
        data = self.driver.embeddings(input_ids.detach().cpu().numpy())
        width = self.metadata["config"]["hidden_size"]
        values = torch.from_numpy(
            np.frombuffer(data, dtype="<f4").copy().reshape(-1, width)
        )
        self.deepstack = None
        if multimodal_embeddings:
            values, self.deepstack = self.media.merge(
                input_ids, values, multimodal_embeddings, is_multimodal, self.config
            )
        return values.reshape(*input_ids.shape, width)

    def get_mrope_input_positions(self, input_tokens, mm_features):
        return multimodal_positions(input_tokens, mm_features, self.config)

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        capture_layer_indices=None,
        return_hidden_states=False,
        **kwargs,
    ):
        if intermediate_tensors is not None:
            raise ValueError("Thinker device owns pipeline state")
        positions = positions.detach().cpu().numpy()
        if positions.ndim == 1:
            positions = np.repeat(positions[None], 3, axis=0)
        if int(positions[0, 0]) == 0:
            self.next_position = 0
        tokens = None if input_ids is None else input_ids.detach().cpu().numpy()
        embeds = None if inputs_embeds is None else inputs_embeds.detach().cpu().numpy()
        features = None if self.deepstack is None else self.deepstack.numpy()
        capture = () if capture_layer_indices is None else tuple(capture_layer_indices)
        result = self.driver.forward(
            tokens, positions, self.next_position, embeds, features, capture
        )
        self.next_position += result.shape[0]
        self.deepstack = None
        if not np.isfinite(result).all():
            raise RuntimeError("non-finite Thinker hidden states")
        hidden = torch.from_numpy(result)
        if return_hidden_states:
            return hidden, {
                "hidden_states": {
                    "layers": {
                        index: torch.from_numpy(value)
                        for index, value in self.driver.captures.items()
                    }
                }
            }
        return hidden

    def compute_logits(self, hidden_states):
        logits = self.driver.logits(hidden_states.detach().cpu().numpy())
        if not np.isfinite(logits).all():
            raise RuntimeError("non-finite Thinker logits")
        return torch.from_numpy(logits)
