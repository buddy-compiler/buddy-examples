from torch import nn
from torch.nn import functional as F


class Convolution(nn.Module):
    def __init__(self, weight, bias, channels, kernel, *, dilation=1, stride=1):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.channels = channels
        self.kernel = kernel
        self.dilation = dilation
        self.stride = stride

    def forward(self, hidden, output_rows=None):
        effective = (self.kernel - 1) * self.dilation + 1
        extra = (-hidden.shape[-1]) % self.stride
        patches = F.pad(hidden, (effective - self.stride, extra)).unfold(
            2, effective, self.stride
        )
        patches = patches[..., :: self.dilation]
        if output_rows is not None:
            patches = patches[:, :, output_rows[0] : output_rows[1], :]
        length = patches.shape[2]
        patches = patches.permute(0, 2, 1, 3).reshape(-1, self.channels * self.kernel)
        patches = F.pad(patches, (0, self.weight.shape[-1] - patches.shape[-1]))
        result = F.linear(patches, self.weight) + self.bias
        return result.reshape(1, length, -1).transpose(1, 2)


class Transpose(nn.Module):
    def __init__(self, weight, bias, channels, output_channels, kernel, stride):
        super().__init__()
        if kernel not in (stride, 2 * stride):
            raise ValueError(
                "Code2Wav transpose kernels require one or two stride widths"
            )
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.channels = channels
        self.output_channels = output_channels
        self.kernel = kernel
        self.stride = stride

    def forward(self, hidden):
        length = hidden.shape[-1]
        patches = hidden.transpose(1, 2).reshape(-1, self.channels)
        patches = F.pad(patches, (0, self.weight.shape[-1] - self.channels))
        projected = F.linear(patches, self.weight)
        projected = projected.reshape(
            1, length, self.output_channels, self.kernel
        ).permute(0, 2, 1, 3)
        if self.kernel == self.stride:
            result = projected.reshape(1, self.output_channels, length * self.stride)
        else:
            first = projected[..., : self.stride].reshape(
                1, self.output_channels, length * self.stride
            )
            second = projected[..., self.stride :].reshape(
                1, self.output_channels, length * self.stride
            )
            result = (F.pad(first, (0, self.stride)) + F.pad(second, (self.stride, 0)))[
                ..., self.stride : -self.stride
            ]
        return result + self.bias[None, :, None]


class Snake(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = nn.Parameter(alpha)
        self.beta = nn.Parameter(beta)

    def forward(self, hidden):
        alpha = self.alpha.exp()[None, :, None]
        beta = self.beta.exp()[None, :, None]
        return hidden + (hidden * alpha).sin().square() / (beta + 1e-9)


class Residual(nn.Module):
    def __init__(self, activation0, convolution0, activation1, convolution1):
        super().__init__()
        self.activation0 = activation0
        self.convolution0 = convolution0
        self.activation1 = activation1
        self.convolution1 = convolution1

    def forward(self, hidden):
        first = self.convolution0(self.activation0(hidden))
        return hidden + self.convolution1(self.activation1(first))


class Upsample(nn.Module):
    def __init__(self, activation, convolution):
        super().__init__()
        self.activation = activation
        self.convolution = convolution

    def forward(self, hidden):
        return self.convolution(self.activation(hidden))


class ConvNeXt(nn.Module):
    def __init__(self, convolution, tensors):
        super().__init__()
        self.convolution = convolution
        self.weights = nn.ParameterDict(
            {k: nn.Parameter(v) for k, v in tensors.items()}
        )

    def forward(self, hidden):
        value = self.convolution(hidden).transpose(1, 2)
        value = F.layer_norm(
            value,
            (value.shape[-1],),
            self.weights["norm"],
            self.weights["norm_bias"],
            1e-6,
        )
        value = (
            F.linear(value.reshape(-1, value.shape[-1]), self.weights["up"])
            + self.weights["up_bias"]
        )
        value = (
            F.linear(F.gelu(value), self.weights["down"]) + self.weights["down_bias"]
        )
        value = value.reshape(1, -1, hidden.shape[1]) * self.weights["scale"]
        return hidden + value.transpose(1, 2)
