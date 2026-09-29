import numpy as np
import torch
from torch import nn
from buddy.compiler.graph.transform.quantization.activation import calibrate_layers
from stack.compiler.quant.importer import quantize_model_graph
from buddy.compiler.graph.transform.quantization.symmetric import quantize_symmetric
from ..permute.layout import reorder

FORMAT = "int8_per_channel_smooth_dynamic"
STAGES = {"attention", "ffn"}


class ScaledLinear(nn.Module):
    def __init__(self, linear, scale):
        super().__init__()
        self.bias = linear.bias
        linear.bias = None
        self.linear = linear
        self.register_buffer("input_scale", scale)
        with torch.no_grad():
            self.linear.weight.mul_(scale)

    def forward(self, values):
        values = values / self.input_scale
        amplitude = values.abs().amax(-1, keepdim=True).clamp_min(1e-5)
        result = self.linear(values / amplitude) * amplitude
        if self.bias is not None:
            result = result + self.bias
        return result


@torch.no_grad()
def prepare(stage, samples):
    ranges, hooks = {}, []
    for name, module in stage.named_children():
        if isinstance(module, nn.Linear):
            ranges[name] = torch.zeros(module.in_features)

            def record(layer, inputs, name=name):
                maximum = inputs[0].abs().reshape(-1, layer.in_features).amax(0)
                ranges[name] = torch.maximum(ranges[name], maximum)

            hooks.append(module.register_forward_pre_hook(record))
    try:
        for sample in samples:
            stage(**sample)
    finally:
        for hook in hooks:
            hook.remove()
    for name, maximum in ranges.items():
        linear = getattr(stage, name)
        weight_max = linear.weight.abs().amax(0).clamp_min(1e-5)
        # (x / scale) @ (weight * scale).T preserves the float computation.
        scale = (maximum.clamp_min(1e-5) / weight_max).sqrt().clamp_min(1e-5)
        setattr(stage, name, ScaledLinear(linear, scale))


PREPARE = prepare


def apply(graph, params, names, output, name, packer, stage, samples):
    calibration = calibrate_layers(stage, samples[0])
    for sample in samples[1:]:
        candidate = calibrate_layers(stage, sample)
        for key, records in calibration.items():
            calibration[key] = [
                tuple(np.maximum(a, b) for a, b in zip(left, right, strict=True))
                for left, right in zip(records, candidate[key], strict=True)
            ]
    scales = {
        key: quantize_symmetric(value.detach().numpy(), [0])[1]
        for key, value in zip(names, params)
        if key.endswith("weight") and value.ndim == 2
    }
    quantize_model_graph(
        graph,
        params,
        names,
        output,
        name,
        calibration,
        packer,
        reorder=reorder,
        weight_scales=scales,
    )
