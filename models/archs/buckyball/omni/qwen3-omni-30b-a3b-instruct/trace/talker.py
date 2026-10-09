import importlib

Checkpoint = importlib.import_module(
    "stack.models.models.qwen3-omni-30b-a3b-instruct.weights"
).Checkpoint
speech = importlib.import_module(
    "stack.models.models.qwen3-omni-30b-a3b-instruct.talker"
)
moe = importlib.import_module("stack.models.models.qwen3-omni-30b-a3b-instruct.moe")
attention = importlib.import_module(
    "stack.models.models.qwen3-omni-30b-a3b-instruct.attention"
)
projections = importlib.import_module(
    "stack.models.models.qwen3-omni-30b-a3b-instruct.projections"
)


class Weights(Checkpoint):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        self.config = self.metadata["talker_config"]
        self.text = self.config["text_config"]
        self.predictor = self.config["code_predictor_config"]
        if not self.text["norm_topk_prob"]:
            raise ValueError("Talker requires normalized top-K routing")

    def router(self, layer):
        prefix = f"talker.model.layers.{layer}"
        return speech.Router(
            self.tensor(f"{prefix}.post_attention_layernorm.weight"),
            self.tensor(f"{prefix}.mlp.gate.weight"),
            self.tensor(f"{prefix}.mlp.shared_expert_gate.weight"),
            self.text,
        ).eval()

    def expert(self, layer, expert):
        prefix = f"talker.model.layers.{layer}.mlp.experts.{expert}"
        return moe.Expert(
            self.tensor(f"{prefix}.gate_proj.weight"),
            self.tensor(f"{prefix}.up_proj.weight"),
            self.tensor(f"{prefix}.down_proj.weight"),
        ).eval()

    def shared(self, layer):
        prefix = f"talker.model.layers.{layer}.mlp.shared_expert"
        return moe.Expert(
            self.tensor(f"{prefix}.gate_proj.weight"),
            self.tensor(f"{prefix}.up_proj.weight"),
            self.tensor(f"{prefix}.down_proj.weight"),
        ).eval()

    def attention(self, layer, *, predictor=False):
        prefix = (
            f"talker.code_predictor.model.layers.{layer}"
            if predictor
            else f"talker.model.layers.{layer}"
        )
        config = dict(self.predictor if predictor else self.text)
        if predictor:
            config["rope_scaling"] = {
                "rope_type": "default",
                "mrope_interleaved": True,
                "mrope_section": [64, 0, 0],
            }
        else:
            config["rope_scaling"] = {
                **config["rope_scaling"],
                "mrope_interleaved": config["rope_scaling"]["interleaved"],
            }
        tensors = {"norm": self.tensor(f"{prefix}.input_layernorm.weight")}
        for name in ("q_norm", "k_norm"):
            tensors[name] = self.tensor(f"{prefix}.self_attn.{name}.weight")
        for name in ("q", "k", "v", "o"):
            tensors[name] = self.tensor(f"{prefix}.self_attn.{name}_proj.weight")
        return attention.Attention(tensors, config, 1).eval()

    def dense(self, layer):
        prefix = f"talker.code_predictor.model.layers.{layer}"
        return speech.Dense(
            self.tensor(f"{prefix}.post_attention_layernorm.weight"),
            self.tensor(f"{prefix}.mlp.gate_proj.weight"),
            self.tensor(f"{prefix}.mlp.up_proj.weight"),
            self.tensor(f"{prefix}.mlp.down_proj.weight"),
            self.predictor["rms_norm_eps"],
        ).eval()

    def resize(self, kind):
        prefix = f"talker.{kind}_projection"
        return speech.Resize(
            self.tensor(f"{prefix}.linear_fc1.weight"),
            self.tensor(f"{prefix}.linear_fc1.bias"),
            self.tensor(f"{prefix}.linear_fc2.weight"),
            self.tensor(f"{prefix}.linear_fc2.bias"),
        ).eval()

    def norm(self, *, predictor=False):
        prefix = "talker.code_predictor.model" if predictor else "talker.model"
        config = self.predictor if predictor else self.text
        return projections.Norm(
            self.tensor(f"{prefix}.norm.weight"), config["rms_norm_eps"]
        ).eval()

    def output(self, group=-1):
        name = (
            "talker.codec_head.weight"
            if group == -1
            else f"talker.code_predictor.lm_head.{group}.weight"
        )
        return projections.Output(self.tensor(name)).eval()
