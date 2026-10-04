import importlib

import torch
from torch.nn import functional as F

Checkpoint = importlib.import_module(
    "stack.models.models.qwen3-omni-30b-a3b-instruct.weights"
).Checkpoint
convolution = importlib.import_module(
    "stack.models.models.qwen3-omni-30b-a3b-instruct.wave_conv"
)
transformer = importlib.import_module(
    "stack.models.models.qwen3-omni-30b-a3b-instruct.wave_transformer"
)
projections = importlib.import_module(
    "stack.models.models.qwen3-omni-30b-a3b-instruct.projections"
)


class SpanResidual(convolution.Residual):
    def __init__(self, residual, owned_rows):
        super().__init__(
            residual.activation0,
            residual.convolution0,
            residual.activation1,
            residual.convolution1,
        )
        self.owned_rows = owned_rows
        self.second_halo = (self.convolution1.kernel - 1) * self.convolution1.dilation
        self.halo = (
            self.convolution0.kernel - 1
        ) * self.convolution0.dilation + self.second_halo

    def forward(self, hidden):
        first = self.convolution0(
            self.activation0(hidden),
            (self.halo - self.second_halo, self.halo + self.owned_rows),
        )
        value = self.convolution1(
            self.activation1(first),
            (self.second_halo, self.second_halo + self.owned_rows),
        )
        return hidden[..., self.halo : self.halo + self.owned_rows] + value


class Weights(Checkpoint):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        self.config = self.metadata["code2wav_config"]

    def conv(self, prefix, *, channels, dilation=1, groups=1):
        weight = self.tensor(f"{prefix}.conv.weight")
        outputs, grouped_channels, kernel = weight.shape
        dense = torch.zeros(outputs, channels * kernel)
        for row in range(outputs):
            first = (row // (outputs // groups)) * grouped_channels * kernel
            dense[row, first : first + grouped_channels * kernel] = weight[
                row
            ].flatten()
        dense = F.pad(dense, (0, (-dense.shape[1]) % 32))
        return convolution.Convolution(
            dense,
            self.tensor(f"{prefix}.conv.bias"),
            channels,
            kernel,
            dilation=dilation,
        )

    def transpose(self, prefix, stride):
        weight = self.tensor(f"{prefix}.conv.weight")
        channels, outputs, kernel = weight.shape
        matrix = weight.permute(1, 2, 0).reshape(outputs * kernel, channels)
        matrix = F.pad(matrix, (0, (-channels) % 32))
        return convolution.Transpose(
            matrix,
            self.tensor(f"{prefix}.conv.bias"),
            channels,
            outputs,
            kernel,
            stride,
        )

    def snake(self, prefix):
        return convolution.Snake(
            self.tensor(f"{prefix}.alpha"), self.tensor(f"{prefix}.beta")
        )

    def attention(self, layer):
        prefix = f"code2wav.pre_transformer.layers.{layer}"
        tensors = {
            "norm": self.tensor(f"{prefix}.input_layernorm.weight"),
            "scale": self.tensor(f"{prefix}.self_attn_layer_scale.scale"),
        }
        for kind in ("q", "k", "v", "o"):
            tensors[kind] = self.tensor(f"{prefix}.self_attn.{kind}_proj.weight")
        return transformer.Attention(tensors, self.config).eval()

    def dense(self, layer):
        prefix = f"code2wav.pre_transformer.layers.{layer}"
        tensors = {
            "norm": self.tensor(f"{prefix}.post_attention_layernorm.weight"),
            "scale": self.tensor(f"{prefix}.mlp_layer_scale.scale"),
        }
        for kind in ("gate", "up", "down"):
            tensors[kind] = self.tensor(f"{prefix}.mlp.{kind}_proj.weight")
        return transformer.Dense(tensors, self.config["rms_norm_eps"]).eval()

    def norm(self):
        return projections.Norm(
            self.tensor("code2wav.pre_transformer.norm.weight"),
            self.config["rms_norm_eps"],
        ).eval()

    def upsample(self, index):
        return self.transpose(
            f"code2wav.upsample.{index}.0", self.config["upsampling_ratios"][index]
        ).eval()

    def convnext(self, index):
        prefix = f"code2wav.upsample.{index}.1"
        names = {
            "norm": "norm.weight",
            "norm_bias": "norm.bias",
            "scale": "gamma",
            "up": "pwconv1.weight",
            "up_bias": "pwconv1.bias",
            "down": "pwconv2.weight",
            "down_bias": "pwconv2.bias",
        }
        return convolution.ConvNeXt(
            self.conv(
                f"{prefix}.dwconv",
                channels=self.config["hidden_size"],
                groups=self.config["hidden_size"],
            ),
            {k: self.tensor(f"{prefix}.{v}") for k, v in names.items()},
        ).eval()

    def decoder_input(self):
        return self.conv(
            "code2wav.decoder.0", channels=self.config["hidden_size"]
        ).eval()

    def decoder_upsample(self, index):
        prefix = f"code2wav.decoder.{index + 1}.block"
        return convolution.Upsample(
            self.snake(f"{prefix}.0"),
            self.transpose(f"{prefix}.1", self.config["upsample_rates"][index]),
        ).eval()

    def residual(self, index, unit):
        prefix = f"code2wav.decoder.{index + 1}.block.{unit + 2}"
        channels = self.config["decoder_dim"] // 2 ** (index + 1)
        return convolution.Residual(
            self.snake(f"{prefix}.act1"),
            self.conv(f"{prefix}.conv1", channels=channels, dilation=(1, 3, 9)[unit]),
            self.snake(f"{prefix}.act2"),
            self.conv(f"{prefix}.conv2", channels=channels),
        ).eval()

    def output(self):
        index = len(self.config["upsample_rates"]) + 1
        channels = self.config["decoder_dim"] // 2 ** len(self.config["upsample_rates"])
        return convolution.Upsample(
            self.snake(f"code2wav.decoder.{index}"),
            self.conv(f"code2wav.decoder.{index + 1}", channels=channels),
        ).eval()
