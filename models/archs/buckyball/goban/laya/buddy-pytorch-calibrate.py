from importlib import import_module
from pathlib import Path
import json
import math

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from compiler.frontend.quant import quantize, quantize_bias
from frontend.quant.graph import smooth_inputs

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
decisions = import_module("models.models.laya.data")
laya = import_module("models.models.laya.model")
# The recipe runs the checkpoint fine-tuned on typed decisions.
laya.checkpoint = laya.checkpoint / "typed-decisions"


class Products(TorchFunctionMode):
    """Calls `linear(input, weight)` on the input of every Linear and
    `product(index, operands)` on every attention product, numbered in the
    order the model runs them, and computes with what they return. The
    attention of the head runs as the imported graph runs it: both operands
    of the scores take the square root of the softmax scale. A product of
    one column (the rotary frequencies) is a broadcast multiply in the graph,
    not a matrix product. The attention of the head reaches the mode as one
    multi_head_attention_forward call: its body runs under the mode, without
    dispatching to it again, so that its Linears and products reach it."""

    def __init__(self, linear, product):
        super().__init__()
        self.linear = linear
        self.product = product
        self.index = 0

    def __torch_function__(self, function, types, args=(), kwargs=None):
        name = getattr(function, "__name__", None)
        if name == "linear":
            args = (self.linear(args[0], args[1]), *args[1:])
        elif name == "matmul" and args[0].shape[-1] > 1:
            args = self.product(self.index, args)
            self.index += 1
        elif name == "scaled_dot_product_attention":
            query, key, value, mask = args[:4]
            root = 1 / math.sqrt(math.sqrt(query.shape[-1]))
            query, key = self.product(
                self.index, (query * root, key.transpose(-2, -1) * root)
            )
            weights = torch.softmax(query @ key + mask, -1)
            weights, value = self.product(self.index + 1, (weights, value))
            self.index += 2
            return weights @ value
        elif name == "multi_head_attention_forward":
            dispatch = F.has_torch_function
            F.has_torch_function = lambda operands: False
            with self:
                result = function(*args, **(kwargs or {}))
            F.has_torch_function = dispatch
            return result
        return function(*args, **(kwargs or {}))


# ===----------------------------------------------------------------------===
# Record the largest magnitude of every input channel of every Linear and of
# both operands of every attention product over the questions of 64 records.
# ===----------------------------------------------------------------------===
model = laya.load()
weights = {parameter: name for name, parameter in model.named_parameters()}
inputs, _ = decisions.load(0, 64)
maxima = {}
channel_maxima = {}


def record(name, value):
    maxima[name] = max(maxima.get(name, 0.0), value.abs().max().item())


def record_linear(x, weight):
    name = weights[weight].rsplit(".", 1)[0]
    maximum = x.abs().amax(list(range(x.ndim - 1)))
    channel_maxima[name] = torch.maximum(channel_maxima.get(name, maximum), maximum)
    return x


def record_product(index, operands):
    record(f"matmul.{index}.0", operands[0])
    record(f"matmul.{index}.1", operands[1])
    return operands


with torch.no_grad():
    for batch in zip(*(value.split(16) for value in inputs)):
        with Products(record_linear, record_product):
            model(*batch)

# ===----------------------------------------------------------------------===
# Smooth the inputs of the encoder Wqkv and Wi through the LayerNorm that
# produces them (the first layer reads the embeddings without a norm), and
# the inputs of mlp.Wo through the gate half of Wi: the producer divides
# every channel by s, the Linear weight takes s. s = sqrt(a / w) from the
# largest input magnitude a and weight magnitude w of the channel.
# Every Linear then converts its whole input with one multiplier, from its
# largest (smoothed) magnitude.
# ===----------------------------------------------------------------------===
smooth = {}
for name, module in model.named_modules():
    if name.endswith(("mlp.Wi", "mlp.Wo")) or (
        name.endswith("attn.Wqkv") and name != "encoder.model.layers.0.attn.Wqkv"
    ):
        a = channel_maxima[name].clamp(min=1e-5)
        w = module.weight.detach().abs().amax(0).clamp(min=1e-5)
        smooth[name] = (a / w).sqrt().tolist()
for name, maximum in channel_maxima.items():
    if name in smooth:
        maximum = maximum / torch.tensor(smooth[name])
    maxima[name] = maximum.max().item()

# ===----------------------------------------------------------------------===
# Evaluate the quantization in software on the questions of 200 other
# records: the scales are rejected when the decision accuracy against the
# gold answers drops more than 1 point below FP32. Every Linear reads its
# weight with one scale per output channel and its (smoothed) input with
# one scale;
# every attention product reads both operands with one scale each.
# ===----------------------------------------------------------------------===
tests, labels = decisions.load(200, 200)


def evaluate(model, linear=None, product=None):
    """The decision accuracy and the logits of the markers."""
    logits = []
    with torch.no_grad():
        for batch in zip(*(value.split(16) for value in tests)):
            if linear is None:
                logits.append(model(*batch)[0])
            else:
                with Products(linear, product):
                    logits.append(model(*batch)[0])
    logits = torch.cat(logits)
    return (logits.argmax(1) == labels).float().mean().item() * 100, logits


def fake_linear(x, weight):
    scale = torch.tensor(maxima[weights[weight].rsplit(".", 1)[0]] / 127)
    return quantize(x, 1 / scale) * scale


def fake_product(index, operands):
    scales = [torch.tensor(maxima[f"matmul.{index}.{side}"] / 127) for side in (0, 1)]
    return tuple(
        quantize(operand, 1 / scale) * scale for operand, scale in zip(operands, scales)
    )


fp32, reference = evaluate(model)
quantized = laya.load()
smooth_inputs(quantized, smooth)
weights = {parameter: name for name, parameter in quantized.named_parameters()}
parameters = dict(quantized.named_parameters())
for name, parameter in parameters.items():
    module = name.rsplit(".", 1)[0]
    if not name.endswith("weight") or module not in maxima:
        continue
    step = parameter.detach().abs().amax(1, keepdim=True) / 127
    parameter.data = quantize(parameter.detach(), 1 / step) * step
    bias = parameters.get(name[: -len("weight")] + "bias")
    if bias is not None:
        bias_step = step.reshape(-1) * torch.tensor(maxima[module] / 127)
        bias.data = quantize_bias(bias.detach(), bias_step) * bias_step
int8, logits = evaluate(quantized, fake_linear, fake_product)
# The logits of the markers that hold an option.
valid = tests[3]
error = ((logits - reference)[valid].norm() / reference[valid].norm()).item()
print(
    f"FP32 accuracy {fp32:.2f}%, INT8 accuracy {int8:.2f}% on {len(labels)} "
    f"decisions, marker logits relative L2 error {error:.4f}"
)
if fp32 - int8 > 1:
    raise RuntimeError(
        f"INT8 accuracy {int8:.2f}% is more than 1 below FP32 {fp32:.2f}%"
    )
calibration = {**maxima, "smooth": smooth}
(output / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
