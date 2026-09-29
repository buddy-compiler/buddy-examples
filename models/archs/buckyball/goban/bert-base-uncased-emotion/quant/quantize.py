from buddy.compiler.graph.transform.quantization.activation import calibrate_layers
from buddy.compiler.graph.transform.quantization.symmetric import quantize_symmetric
from stack.compiler.quant.importer import quantize_model_graph
from ..permute.layout import reorder


def quantize(graph, params, names, output, name, calibration, rax_pack, *, assets):
    # This design uses symmetric INT8 weights with one scale per output channel.
    weight_scales = {}
    for parameter_name, parameter in zip(names, params):
        if parameter_name.endswith(".weight") and parameter.ndim >= 2:
            _, scales = quantize_symmetric(parameter.detach().numpy(), [0])
            weight_scales[parameter_name] = scales
    quantize_model_graph(graph, params, names, output, name, calibration, rax_pack,
                         reorder=reorder, weight_scales=weight_scales, assets=assets)


def prepare(model, tokenizer, sequence_length):
    import torch
    texts = ["I feel happy today.", "I am very sad.", "This makes me angry.",
             "I am afraid.", "I love you.", "What a surprise!"]
    calibration_inputs = dict(tokenizer(texts, padding="max_length", truncation=True,
                                        max_length=sequence_length, return_tensors="pt"))
    positions = torch.arange(sequence_length).reshape(1, -1)
    calibration_inputs["position_ids"] = positions.expand_as(calibration_inputs["input_ids"])
    calibration = calibrate_layers(model, calibration_inputs)
    inputs = {name: value[:1] for name, value in calibration_inputs.items()}
    inputs["position_ids"] = positions
    return inputs, calibration
