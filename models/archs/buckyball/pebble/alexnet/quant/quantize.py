from buddy.compiler.graph.transform.quantization.activation import calibrate_layers
from buddy.compiler.graph.transform.quantization.symmetric import quantize_symmetric
from stack.compiler.quant.importer import quantize_model_graph
from ..permute.layout import reorder, pool_result, projection_output
from ..lowering import ActivationBoundary, PaddedOutputLinear, UnfoldConv
from ..trace.grouping import partition


def quantize(graph, params, names, output, name, calibration, rax_pack):
    # This design uses symmetric INT8 weights with one scale per output channel.
    weight_scales = {}
    for parameter_name, parameter in zip(names, params):
        if parameter_name.endswith(".weight") and parameter.ndim >= 2:
            _, scales = quantize_symmetric(parameter.detach().numpy(), [0])
            weight_scales[parameter_name] = scales
    quantize_model_graph(graph, params, names, output, name, calibration, rax_pack,
                         reorder=reorder, weight_scales=weight_scales)
    graph._ops_registry["MaxPool2dOp"] = pool_result
    graph._ops_registry["PermuteOp"] = projection_output
    partition(graph)


def prepare(model, inputs):
    import torch
    # The first 11x11 convolution exceeds the Im2col bank footprint. The
    # existing MatMul lowering splits its K dimension and accumulates INT32.
    model.features[0] = UnfoldConv(model.features[0])
    # At the compiled 224x224 input, the last pool already produces 6x6.
    # Remove the identity adaptive pool so conv/pool regions can stay fused.
    model.avgpool = torch.nn.Identity()
    # Evaluation dropout is an identity; avoid its cloned activation buffers.
    model.classifier[0] = torch.nn.Identity()
    model.classifier[3] = torch.nn.Identity()
    # Limit recursive tile recomputation across the four-convolution chain.
    model.features[7] = ActivationBoundary()
    model.features[9] = ActivationBoundary()
    model.classifier[2] = ActivationBoundary()
    model.classifier[5] = ActivationBoundary()
    model.classifier[6] = PaddedOutputLinear(model.classifier[6])
    return calibrate_layers(model, inputs)
