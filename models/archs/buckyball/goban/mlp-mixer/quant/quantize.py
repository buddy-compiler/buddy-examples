from buddy.compiler.graph.transform.quantization.activation import calibrate_layers
from buddy.compiler.graph.transform.quantization.symmetric import quantize_symmetric
from stack.compiler.quant.importer import quantize_model_graph
from ..permute.layout import reorder


def prepare(model, data):
    return calibrate_layers(model, data)


def quantize(graph, params, names, output, name, calibration, rax_pack):
    scales = {}
    for parameter_name, parameter in zip(names, params):
        if parameter_name.endswith(".weight") and parameter.ndim == 2:
            _, scales[parameter_name] = quantize_symmetric(parameter.detach().numpy(), [0])
    quantize_model_graph(graph, params, names, output, name, calibration, rax_pack,
                         reorder=reorder, weight_scales=scales)
