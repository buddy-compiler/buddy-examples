import operator

import torch
import torch.nn.functional as F
from compiler.frontend.quant import quantize, quantize_bias

Q = 1 << 24


def round_div(value, denominator):
    absolute = value.abs()
    quotient = torch.div(absolute, denominator, rounding_mode="floor")
    remainder = absolute - quotient * denominator
    rounded = quotient + (
        (remainder * 2 > denominator)
        | ((remainder * 2 == denominator) & ((quotient & 1) != 0))
    )
    return torch.where(value < 0, -rounded, rounded)


class FixedGraph(torch.fx.Interpreter):
    def __init__(self, model, calibration):
        layers = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
        }
        for name, module in layers.items():
            entry = calibration[name]
            smooth = torch.tensor(entry["smooth"])
            step_input = torch.tensor(entry["clip"] / 127 if entry["clip"] else 1.0)
            shape = (
                (1, -1)
                if isinstance(module, torch.nn.Linear)
                else (-1, 1, 1, 1) if module.groups > 1 else (1, -1, 1, 1)
            )
            weight = module.weight.detach() * smooth.reshape(shape)
            maximum = weight.abs().amax(list(range(1, weight.ndim)), keepdim=True)
            floor = module.bias.detach().abs().reshape(maximum.shape) / (
                step_input * 2**30
            )
            step = torch.maximum(maximum / 127, floor)
            step = torch.where(step == 0, 1.0, step)
            module.weight.data = quantize(weight, 1 / step).float()
            bias_step = step.reshape(-1) * step_input
            module.bias.data = quantize_bias(module.bias.detach(), bias_step).float()
            module.input_multiplier = 1 / (step_input * smooth)
            module.output_step = bias_step
            mantissa, exponent = torch.frexp(bias_step.double() * (1 << 24))
            module.fixed_multiplier = torch.round(mantissa * (1 << 30)).long()
            module.fixed_shift = 30 - exponent.long()
            assert torch.all(module.fixed_shift >= 0)
            module.fixed_multiplier = torch.where(
                module.fixed_shift >= 62, 0, module.fixed_multiplier
            )
            module.fixed_shift = torch.where(
                module.fixed_shift >= 62, 0, module.fixed_shift
            )
        self.last_linear = [
            name for name, layer in layers.items() if isinstance(layer, torch.nn.Linear)
        ][-1]
        self.clipped = 0
        super().__init__(torch.fx.symbolic_trace(model))

    def pool(self, value):
        return round_div(
            value.sum((2, 3), keepdim=True), value.shape[2] * value.shape[3]
        )

    def call_module(self, target, args, kwargs):
        module = self.fetch_attr(target)
        x = args[0]
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            shape = [1, -1] + [1] * (x.ndim - 2)
            multiplier = module.input_multiplier.reshape(shape)
            if x.dtype == torch.int64:
                multiplier = multiplier * (2.0**-24)
            codes = quantize(x.float(), multiplier).float()
            if isinstance(module, torch.nn.Conv2d):
                result = F.conv2d(
                    codes,
                    module.weight,
                    None,
                    module.stride,
                    module.padding,
                    module.dilation,
                    module.groups,
                )
            else:
                result = F.linear(codes, module.weight)
            assert result.abs().max() < (1 << 24)
            result = result.long() + module.bias.long().reshape(shape)
            if target == self.last_linear:
                return result.float() * module.output_step.reshape(shape)
            value = round_div(
                result * module.fixed_multiplier.reshape(shape),
                1 << module.fixed_shift.reshape(shape),
            )
            self.clipped += ((value < -(1 << 31)) | (value >= (1 << 31))).sum().item()
            return value.clamp(-(1 << 31), (1 << 31) - 1)
        if isinstance(module, torch.nn.Hardswish):
            middle = round_div(x * (x + 3 * Q), 6 * Q)
            return torch.where(x <= -3 * Q, 0, torch.where(x >= 3 * Q, x, middle))
        if isinstance(module, torch.nn.Hardsigmoid):
            return round_div((x + 3 * Q).clamp(0, 6 * Q), 6)
        if isinstance(module, torch.nn.ReLU):
            return x.clamp(min=0)
        if isinstance(module, (torch.nn.Identity, torch.nn.Dropout)):
            return x
        if isinstance(module, torch.nn.AdaptiveAvgPool2d):
            assert module.output_size in (1, (1, 1), [1, 1])
            return self.pool(x)
        raise TypeError(type(module))

    def call_function(self, target, args, kwargs):
        if target == operator.add:
            return (args[0] + args[1]).clamp(-(1 << 31), (1 << 31) - 1)
        if target == operator.mul:
            return round_div(args[0] * args[1], Q).clamp(-(1 << 31), (1 << 31) - 1)
        if target == F.adaptive_avg_pool2d:
            assert args[1] in (1, (1, 1), [1, 1])
            return self.pool(args[0])
        if target == torch.flatten:
            return torch.flatten(*args, **kwargs)
        raise TypeError(target)
