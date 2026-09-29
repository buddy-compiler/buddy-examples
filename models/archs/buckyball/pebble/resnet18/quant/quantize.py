from buddy.compiler.graph.transform.quantization.activation import calibrate_layers
from buddy.compiler.graph.transform.quantization.symmetric import quantize_symmetric
from stack.compiler.quant.importer import quantize_model_graph
from ..permute.layout import reorder


def quantize(graph, params, names, output, name, calibration, rax_pack):
    # This design uses symmetric INT8 weights with one scale per output channel.
    weight_scales = {}
    for parameter_name, parameter in zip(names, params):
        if parameter_name.endswith(".weight") and parameter.ndim >= 2:
            _, scales = quantize_symmetric(parameter.detach().numpy(), [0])
            weight_scales[parameter_name] = scales
    quantize_model_graph(graph, params, names, output, name, calibration, rax_pack,
                         reorder=reorder, weight_scales=weight_scales)


def prepare(model):
    import torch
    inputs = torch.randn([1, 3, 224, 224])
    return inputs, calibrate_layers(model, inputs)
