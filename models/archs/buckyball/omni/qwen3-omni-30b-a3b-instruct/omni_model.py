import json
from pathlib import Path

import torch
from torch import nn
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerMultiModalProcessor,
    Qwen3OmniMoeThinkerProcessingInfo,
    Qwen3OmniMoeThinkerDummyInputsBuilder,
)
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    Qwen3OmniMoeForConditionalGeneration,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

from .model import Model
from .speech_model import Speech, Vocoder
from .conditioning import Conditioning


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class OmniModel(Conditioning, Qwen3OmniMoeForConditionalGeneration):
    is_attention_free = True

    def __init__(self, *, vllm_config, prefix=""):
        nn.Module.__init__(self)
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.thinker_config = self.config.thinker_config
        self.talker_config = self.config.talker_config
        self.code2wav_config = self.config.code2wav_config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        self.model_stage = vllm_config.model_config.model_stage
        self.have_multimodal_outputs = True
        self.use_async_omni_output = False
        self.has_preprocess = self.model_stage == "talker"
        self.has_postprocess = self.model_stage == "talker"
        self.requires_raw_input_tokens = self.model_stage != "thinker"
        self.is_staged_run = True
        self._returns_tuple = self.model_stage == "thinker"
        self.driver_options = vllm_config.additional_config["execution"]
        self.metadata = json.loads(Path(self.driver_options["metadata"]).read_text())
        self.thinker = self.talker = self.code2wav = None
        if self.model_stage != "talker":
            self.talker_mtp = None
        if self.model_stage == "thinker":
            self.thinker = Model(vllm_config=vllm_config)
            self.model = self.thinker
            self.tts_tokens = torch.tensor(
                [
                    [
                        self.config.tts_bos_token_id,
                        self.config.tts_eos_token_id,
                        self.config.tts_pad_token_id,
                    ]
                ]
            )
        elif self.model_stage == "talker":
            self.talker_mtp_graph_safe = False
            self.omni_pooler_payload_include_hidden = False
            self.gpu_resident_buffer_keys = {
                ("hidden_states", "last"),
                ("hidden_states", "trailing_text"),
                ("embed", "tts_pad_projected"),
                ("codes", "audio"),
            }
            self.enable_update_additional_information = True
            self.set_custom_preprocess(self.talker_preprocess)
            self.set_custom_postprocess(self.talker_postprocess)
        elif self.model_stage == "code2wav":
            self.enable_update_additional_information = True
            self.requires_exact_input_shape = True
        else:
            raise ValueError(f"unknown model stage: {self.model_stage}")
        self.make_empty_intermediate_tensors = lambda *args, **kwargs: None
        from vllm.v1.sample.sampler import Sampler

        self.sampler = Sampler()

    def load_device_weights(self):
        directory = Path(self.driver_options["directory"])
        timeout = self.driver_options["timeout"]
        if self.model_stage == "thinker":
            self.thinker.load_device_weights()
        elif self.model_stage == "talker":
            self.talker = Speech(directory, self.metadata, timeout, self.talker_config)
            self.model = self.talker
            self._init_special_tokens_embeddings()
            self.suppressed_tokens = self._get_talker_suppressed_tokens()
        else:
            self.code2wav = Vocoder(directory, self.metadata, timeout)
            self.model = self.code2wav

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if self.model_stage == "thinker":
            return self.thinker(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                capture_layer_indices=[0, self.talker_config.accept_hidden_layer],
                return_hidden_states=True,
                **kwargs,
            )
        return super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )

    def sample(self, logits, sampling_metadata):
        return self.sampler(logits=logits, sampling_metadata=sampling_metadata)

    def _load_talker_embedding(self):
        return self.talker.embedding

    def get_language_model(self):
        return self.model

    def make_omni_output(self, model_outputs, **kwargs):
        if self.model_stage == "thinker":
            hidden, captured = model_outputs
            embeds = self.thinker.embed_input_ids(self.tts_tokens).chunk(3, dim=1)
            captured["embed"] = {
                name: [value]
                for name, value in zip(
                    ("tts_bos", "tts_eos", "tts_pad"), embeds, strict=True
                )
            }
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs=captured)
        return super().make_omni_output(model_outputs, **kwargs)
