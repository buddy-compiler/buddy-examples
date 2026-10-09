from buddy.compiler.graph.transform.quantization.activation import calibrate_layers
from buddy.compiler.graph.transform.quantization.symmetric import quantize_symmetric
from examples.balls.gemmini.compiler.python.quant import apply
from ..permute.layout import reorder, pool_result


def prepare(model, inputs):
    return calibrate_layers(model, inputs)


def quantize(graph, params, names, output, name, calibration, rax_pack):
    quantized = {
        key: quantize_symmetric(value.detach().numpy(), [])
        for key, value in zip(names, params)
        if value.ndim >= 2
    }
    apply(graph, params, names, output, name, calibration, quantized, rax_pack,
          reorder=reorder, pool_result=pool_result)
