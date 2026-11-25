from importlib import import_module

import torch
from torch.nn.utils.fusion import fuse_conv_bn_eval
from compiler.frontend.quant import quantize, quantize_bias

mobilenet = import_module("models.models.mobilenet-v3-small.model")


def folded():
    """The checkpoint, every batch normalization folded into the convolution
    before it."""
    model = mobilenet.load()
    for block in model.modules():
        if (
            isinstance(block, torch.nn.Sequential)
            and isinstance(block[0], torch.nn.Conv2d)
            and isinstance(block[1], torch.nn.BatchNorm2d)
        ):
            block[0] = fuse_conv_bn_eval(block[0], block[1])
            block[1] = torch.nn.Identity()
    return model


def channel_steps(weight, bias, maximum_input):
    # A folded channel can have near-zero weights and a substantial bias.
    # Bound both encodings: INT8 weights and INT32 bias with room for the dot product.
    input_step = maximum_input / 127 if maximum_input else 1.0
    shape = [-1] + [1] * (weight.ndim - 1)
    weight_step = weight.abs().amax(tuple(range(1, weight.ndim)), keepdim=True) / 127
    bias_bound = (bias.abs() / (input_step * 2**30)).reshape(shape)
    step = torch.maximum(weight_step, bias_bound)
    return torch.where(step == 0, 1.0, step), input_step


def reference(maxima):
    quantized = folded()
    for name, module in quantized.named_modules():
        if name not in maxima:
            continue
        step, scale = channel_steps(
            module.weight.detach(), module.bias.detach(), maxima[name]
        )
        module.weight.data = quantize(module.weight.detach(), 1 / step) * step
        bias_step = step.reshape(-1) * scale
        module.bias.data = quantize_bias(module.bias.detach(), bias_step) * bias_step
        module.register_forward_pre_hook(
            lambda module, inputs, scale=scale: quantize(inputs[0], 1 / scale) * scale
        )
    return quantized
