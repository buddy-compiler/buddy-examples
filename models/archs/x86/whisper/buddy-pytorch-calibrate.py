# ===- buddy-pytorch-calibrate.py -----------------------------------------
#
# Whisper quantization check: the model as the quantized graph computes it
# (Linears and attention products in MXINT8, the 1-D convolutions in INT8),
# rejected when the transcript of the example audio differs from FP32. Every
# activation scale is computed online, so nothing is calibrated:
# calibration.json records the accepted transcript.
#
# ===---------------------------------------------------------------------------

from importlib import import_module
from pathlib import Path
import json

import soundfile
import torch
from torch.overrides import TorchFunctionMode
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from compiler.frontend.quant import dequantize_mxint8, quantize, quantize_mxint8

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
whisper = import_module("models.models.whisper.model")
model = whisper.load()
# The attention runs through explicit products, as the importer.
model.model.config._attn_implementation = "eager"
features = WhisperFeatureExtractor.from_pretrained(whisper.checkpoint)
tokenizer = WhisperTokenizer.from_pretrained(whisper.checkpoint)
# The audio of the Buddy Whisper example, the one the runner transcribes.
example = model_dir.parents[3] / "thirdparty/buddy-mlir/examples/BuddyWhisper/audio.wav"


def mxint8(values, axis):
    """Values through MXINT8 along `axis`, as the graph computes them."""
    return dequantize_mxint8(quantize_mxint8(values, axis), axis)


def tokenwise(x, axis):
    """x as INT8 values with one scale per token: the largest magnitude
    along `axis` over 127."""
    step = x.abs().amax(axis, keepdim=True).clamp(min=1e-30) / 127
    return quantize(x, 1 / step) * step


class Products(TorchFunctionMode):
    """Reads both operands of every attention product through MXINT8 along
    its reduction axis; a product whose K is not a multiple of 32 stays
    FP32."""

    def __torch_function__(self, function, types, args=(), kwargs=None):
        # `a @ b` arrives as torch._C.TensorBase.matmul.
        if (
            getattr(function, "__name__", None) == "matmul"
            and args[0].shape[-1] % 32 == 0
        ):
            args = (mxint8(args[0], -1), mxint8(args[1], -2))
        return function(*args, **(kwargs or {}))


def transcribe(audio, quantized):
    """The greedy tokens of the runner: the whole 448-token window runs for
    every token, the free positions hold the start token."""
    tokens = torch.full((1, 448), 50258)
    count = 0
    with torch.no_grad():
        while count < 447:
            if quantized:
                with Products():
                    logits = model(audio, tokens)
            else:
                logits = model(audio, tokens)
            token = logits[0, count].argmax().item()
            if token == 50257:
                break
            count += 1
            tokens[0, count] = token
    return tokens[0, 1 : count + 1].tolist()


audio, rate = soundfile.read(example)
audio = features(audio, sampling_rate=rate, return_tensors="pt").input_features
fp32 = transcribe(audio, False)

# ===----------------------------------------------------------------------===
# Transcribe again with the quantized model in software: the weights of the
# Linears packed in MXINT8 along K and their inputs quantized online, the
# convolutions with INT8 weights (one scale per output channel) and INT8
# inputs (one online scale per frame), biases FP32. Rejected when the
# transcript differs from FP32.
# ===----------------------------------------------------------------------===
for module in model.modules():
    if isinstance(module, torch.nn.Linear):
        # A new tensor: the output projection shares its weight with the
        # token embedding, which keeps the FP32 values.
        module.weight = torch.nn.Parameter(mxint8(module.weight.detach(), 1))
        module.register_forward_pre_hook(lambda module, args: mxint8(args[0], -1))
    elif isinstance(module, torch.nn.Conv1d):
        weight = module.weight.detach()
        step = weight.abs().amax([1, 2], keepdim=True) / 127
        module.weight.data = quantize(weight, 1 / step) * step
        module.register_forward_pre_hook(lambda module, args: tokenwise(args[0], 1))
quantized = transcribe(audio, True)
print(f"FP32: {tokenizer.decode(fp32)}")
print(f"Quantized: {tokenizer.decode(quantized)}")
if quantized != fp32:
    raise RuntimeError("The quantized transcript differs from FP32")
(output / "calibration.json").write_text(
    json.dumps({"transcript": tokenizer.decode(fp32)}, indent=2) + "\n"
)
