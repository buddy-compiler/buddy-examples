import argparse
import importlib
import json
from pathlib import Path
import sys
import tomllib

import torch

Checkpoint = importlib.import_module(
    "stack.models.models.qwen3-omni-30b-a3b-instruct.weights"
).Checkpoint

stages = importlib.import_module(
    "stack.models.models.qwen3-omni-30b-a3b-instruct.vision"
)


class Weights(Checkpoint):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        self.config = self.metadata["thinker_config"]["vision_config"]
        self.config["num_position_embeddings"] = self.tensor(
            "thinker.visual.pos_embed.weight"
        ).shape[0]

    def patch(self):
        return stages.Patch(
            self.tensor("thinker.visual.patch_embed.proj.weight"),
            self.tensor("thinker.visual.patch_embed.proj.bias"),
            self.tensor("thinker.visual.pos_embed.weight"),
        ).eval()

    def attention(self, layer):
        prefix = f"thinker.visual.blocks.{layer}"
        names = {
            "norm": "norm1.weight",
            "norm_bias": "norm1.bias",
            "qkv": "attn.qkv.weight",
            "qkv_bias": "attn.qkv.bias",
            "projection": "attn.proj.weight",
            "projection_bias": "attn.proj.bias",
        }
        return stages.Attention(
            {k: self.tensor(f"{prefix}.{v}") for k, v in names.items()}, self.config
        ).eval()

    def mlp(self, layer):
        prefix = f"thinker.visual.blocks.{layer}"
        names = {
            "norm": "norm2.weight",
            "norm_bias": "norm2.bias",
            "up": "mlp.linear_fc1.weight",
            "up_bias": "mlp.linear_fc1.bias",
            "down": "mlp.linear_fc2.weight",
            "down_bias": "mlp.linear_fc2.bias",
        }
        tensors = {k: self.tensor(f"{prefix}.{v}") for k, v in names.items()}
        extra = (-tensors["up"].shape[0]) % 32
        tensors["up"] = torch.nn.functional.pad(tensors["up"], (0, 0, 0, extra))
        tensors["up_bias"] = torch.nn.functional.pad(tensors["up_bias"], (0, extra))
        tensors["down"] = torch.nn.functional.pad(tensors["down"], (0, extra))
        return stages.MLP(tensors).eval()

    def merger(self, index):
        prefix = (
            "thinker.visual.merger"
            if index == -1
            else f"thinker.visual.merger_list.{index}"
        )
        names = {
            "norm": "ln_q.weight",
            "norm_bias": "ln_q.bias",
            "up": "mlp.0.weight",
            "up_bias": "mlp.0.bias",
            "down": "mlp.2.weight",
            "down_bias": "mlp.2.bias",
        }
        return stages.Merger(
            {k: self.tensor(f"{prefix}.{v}") for k, v in names.items()},
            self.config,
            index != -1,
        ).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-config", type=Path, required=True)
    parser.add_argument("--compiler-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.compiler_build / "python_packages"))
    from .graph import emit

    settings = tomllib.loads(
        (Path(__file__).parents[1] / "configs/compiler-param.toml").read_text()
    )
    source = Weights(Path(json.loads(args.checkpoint_config.read_text())["path"]))
    config = source.config
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    metadata = {"config": config, "buckets": settings["vision_buckets"], "stages": {}}
    patch_width = (
        config["in_channels"]
        * config["temporal_patch_size"]
        * config["patch_size"] ** 2
    )
    head_dim = config["hidden_size"] // config["num_heads"]
    for count in settings["vision_buckets"]:
        hidden = torch.randn(count, config["hidden_size"])
        plans = [
            (
                "patch",
                source.patch(),
                {
                    "patches": torch.randn(count, patch_width),
                    "indices": torch.zeros(count, 4, dtype=torch.int64),
                    "coefficients": torch.full((count, 4), 0.25),
                },
                "attention",
                {"projection"},
            ),
            (
                "attention",
                source.attention(0),
                {
                    "hidden": hidden,
                    "cosine": torch.ones(count, head_dim),
                    "sine": torch.zeros(count, head_dim),
                    "mask": torch.zeros(count, count, dtype=torch.bool),
                },
                "attention",
                {"weights.qkv", "weights.projection"},
            ),
            (
                "mlp",
                source.mlp(0),
                {"hidden": hidden},
                "ffn",
                {"weights.up", "weights.down"},
            ),
            (
                "merge",
                source.merger(-1),
                {"hidden": hidden},
                "ffn",
                {"weights.up", "weights.down"},
            ),
            (
                "deepstack",
                source.merger(0),
                {"hidden": hidden},
                "ffn",
                {"weights.up", "weights.down"},
            ),
        ]
        for kind, module, inputs, target, quantized in plans:
            name = f"vision_{kind}_{count}"
            metadata["stages"][name] = emit(
                name,
                module,
                inputs,
                target,
                quantized,
                args.output,
                args.compiler_build,
            )
            print(f"captured {name}", flush=True)
    (args.output / "kernels.json").write_text(json.dumps(metadata, indent=2) + "\n")
    constants = {
        "visionWidth": config["hidden_size"],
        "visionHeadDim": head_dim,
        "visionPatchWidth": patch_width,
        "visionOutputWidth": config["out_hidden_size"],
        "visionLayers": config["depth"],
        "visionMerge": config["spatial_merge_size"] ** 2,
    }
    (args.output / "vision-parameters.h").write_text(
        "#pragma once\n#include <cstddef>\n"
        + "".join(f"constexpr size_t {k} = {v};\n" for k, v in constants.items())
        + "constexpr size_t visionBuckets[] = {"
        + ",".join(map(str, settings["vision_buckets"]))
        + "};\n"
        + "constexpr size_t visionDeepstackLayers[] = {"
        + ",".join(map(str, config["deepstack_visual_indexes"]))
        + "};\n"
    )


if __name__ == "__main__":
    main()
