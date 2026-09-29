import torch
from buddy.compiler.graph.transform.quantization.mxfp8 import quantize, dequantize


def linear(value, weight, bias=None):
    value = torch.from_numpy(dequantize(*quantize(value.detach().numpy())))
    weight = torch.from_numpy(dequantize(*quantize(weight.detach().numpy())))
    result = value @ weight.T
    return result if bias is None else result + bias
