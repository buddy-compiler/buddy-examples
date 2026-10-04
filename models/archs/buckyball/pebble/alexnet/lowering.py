import torch
from torch import nn
from torch.nn import functional as F
from .permute.layout import padded_linear_parameters, projection_weights


class ActivationBoundary(nn.Module):
    def forward(self, value):
        # Materialize the layer result so the next layer can reuse the banks.
        return torch.relu(value).clone()


class PaddedOutputLinear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.logical_outputs = linear.out_features
        padded_outputs = (linear.out_features + 15) // 16 * 16
        self.linear = nn.Linear(linear.in_features, padded_outputs)
        weight, bias = padded_linear_parameters(linear.weight.detach(), linear.bias.detach())
        self.linear.weight = nn.Parameter(weight)
        self.linear.bias = nn.Parameter(bias)

    def forward(self, value):
        return self.linear(value)[:, :self.logical_outputs]


class UnfoldConv(nn.Module):
    def __init__(self, convolution):
        super().__init__()
        self.kernel = convolution.kernel_size[0]
        self.stride = convolution.stride[0]
        self.padding = convolution.padding[0]
        self.reduction = (self.kernel * self.kernel + 15) // 16 * 16
        self.bias = convolution.bias
        self.projections = nn.ModuleList()
        for channel, weight in enumerate(projection_weights(convolution.weight.detach())):
            projection = nn.Linear(self.reduction, convolution.out_channels, bias=False)
            projection.weight = nn.Parameter(weight)
            self.projections.append(projection)

    def forward(self, value):
        value = F.pad(value, (self.padding,) * 4)
        result = None
        for channel, projection in enumerate(self.projections):
            patches = value[:, channel:channel + 1].unfold(3, self.kernel, self.stride).unfold(2, self.kernel, self.stride)
            batch, _, height, width, _, _ = patches.shape
            patches = F.pad(patches.reshape(-1, self.kernel * self.kernel),
                            (0, self.reduction - self.kernel * self.kernel))
            partial = projection(patches).clone()
            result = partial if channel == 0 else result + partial
        result = result + self.bias
        return result.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
