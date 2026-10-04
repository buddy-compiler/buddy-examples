import importlib
from pathlib import Path

stages = importlib.import_module("stack.models.models.qwen3-omni-30b-a3b-instruct.moe")


Checkpoint = importlib.import_module(
    "stack.models.models.qwen3-omni-30b-a3b-instruct.weights"
).Checkpoint


class Weights(Checkpoint):
    def __init__(self, checkpoint: Path):
        super().__init__(checkpoint)
        self.config = self.metadata["thinker_config"]["text_config"]

    def router(self, layer):
        prefix = f"thinker.model.layers.{layer}"
        return stages.Router(
            self.tensor(f"{prefix}.post_attention_layernorm.weight"),
            self.tensor(f"{prefix}.mlp.gate.weight"),
            epsilon=self.config["rms_norm_eps"],
            top_k=self.config["num_experts_per_tok"],
            normalize=self.config["norm_topk_prob"],
        ).eval()

    def expert(self, layer, expert, rank, parts):
        prefix = f"thinker.model.layers.{layer}.mlp.experts.{expert}"
        return stages.Expert(
            self.tensor(f"{prefix}.gate_proj.weight", axis=0, rank=rank, parts=parts),
            self.tensor(f"{prefix}.up_proj.weight", axis=0, rank=rank, parts=parts),
            self.tensor(f"{prefix}.down_proj.weight", axis=1, rank=rank, parts=parts),
        ).eval()

    def attention(self, layer, rank, parts):
        config = self.config
        if any(
            config[key] % parts
            for key in ("num_attention_heads", "num_key_value_heads")
        ):
            raise ValueError("attention heads must divide evenly across chips")
        prefix = f"thinker.model.layers.{layer}"
        tensors = {"norm": self.tensor(f"{prefix}.input_layernorm.weight")}
        for name in ("q_norm", "k_norm"):
            tensors[name] = self.tensor(f"{prefix}.self_attn.{name}.weight")
        for name in ("q", "k", "v", "o"):
            tensors[name] = self.tensor(
                f"{prefix}.self_attn.{name}_proj.weight",
                axis=1 if name == "o" else 0,
                rank=rank,
                parts=parts,
            )
        attention = importlib.import_module(
            "stack.models.models.qwen3-omni-30b-a3b-instruct.attention"
        )
        return attention.Attention(tensors, config, parts).eval()

    def embedding(self, rank, parts):
        projections = importlib.import_module(
            "stack.models.models.qwen3-omni-30b-a3b-instruct.projections"
        )
        return projections.Embedding(
            self.tensor(
                "thinker.model.embed_tokens.weight", axis=0, rank=rank, parts=parts
            )
        ).eval()

    def output(self, rank, parts):
        projections = importlib.import_module(
            "stack.models.models.qwen3-omni-30b-a3b-instruct.projections"
        )
        return projections.Output(
            self.tensor("thinker.lm_head.weight", axis=0, rank=rank, parts=parts)
        ).eval()

    def norm(self):
        projections = importlib.import_module(
            "stack.models.models.qwen3-omni-30b-a3b-instruct.projections"
        )
        return projections.Norm(
            self.tensor("thinker.model.norm.weight"), self.config["rms_norm_eps"]
        ).eval()
