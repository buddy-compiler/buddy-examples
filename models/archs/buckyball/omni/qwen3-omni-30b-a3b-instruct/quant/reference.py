from buddy.compiler.graph.transform.quantization.mxfp8 import quantize, dequantize


def linear(value, weight, bias=None):
    value = dequantize(*quantize(value.detach()))
    weight = dequantize(*quantize(weight.detach()))
    result = value @ weight.T
    return result if bias is None else result + bias
