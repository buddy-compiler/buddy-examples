"""The program of pi0-FAST on piplus: the vision encoder once per camera, the
prefill once, then a decoding step per token until the end of the sequence.

`run` computes it in PyTorch as the runner does: it keeps the keys and
values of the generated tokens and the bias of the scores between the steps
and takes the most likely token of every step; it reads the rows of the
embedding table of the tokens for the graphs, which have no table.

`graph` exports one graph as an MLIR module and `prefill` the prefill as a
device program: a function of the control hart that launches vision on a
dense hart once per camera, then the prefill graph as the launches of its
partition (compiler.frontend.partition: the ranks on the action and dense
harts). The decoding step is the device program of its partition
(partition.Device.program). Placement on the harts and the transfers between
them are the device compiler's.

A graph takes its parameters as two packed buffers, FP32 and INT8, in the
order of its parameters, the prefill and the step those of their
partitions."""

import math

import torch

from buddy_mlir import ir
from compiler.frontend.partition import tensor_type
from frontend.graphs import generated, rows


def run(flow, vision, prefill, decode, eos, images, tokens, masks):
    """The tokens up to and with the EOS, and the logits of every step, from
    the prompt `tokens` with the token the model continues after them
    (frame) and their mask."""
    image_tokens = [vision(image) for image in images.unbind(0)]
    logits, position, bias, *cache = prefill(rows(flow, tokens), masks, *image_tokens)
    layers = len(cache) // 2
    prefix = cache[0].shape[2]
    before = [
        torch.zeros(1, 1, generated, cache[0].shape[3]) for _ in range(2 * layers)
    ]
    bias = bias.clone()
    produced, every = [], [logits]
    token = logits[:, -1].argmax(-1, keepdim=True)
    produced.append(int(token))
    for step in range(generated - 1):
        if produced[-1] == eos:
            break
        logits, *own = decode(rows(flow, token), position + step, bias, *cache, *before)
        # The token's keys and values after the generated ones, seen from
        # the next step on.
        for buffer, row in zip(before, own):
            buffer[:, :, step] = row[:, :, 0]
        bias[..., prefix + step] = 0.0
        token = logits[:, -1].argmax(-1, keepdim=True)
        produced.append(int(token))
        every.append(logits)
    return produced, every


def frame(flow, tokens, masks):
    """The prompt and its mask with the token the model continues after
    them (the policy's sample_actions_fast_kv_cache puts it there)."""
    start = torch.full((1, 1), flow._paligemma_tokenizer.bos_token_id, dtype=torch.long)
    return torch.cat([tokens, start], dim=1), torch.cat(
        [masks, torch.ones_like(start)], dim=1
    )


def signature(function):
    """The argument and result types of a func.func, as text."""
    function_type = ir.FunctionType(
        ir.TypeAttr(function.attributes["function_type"]).value
    )
    return [str(t) for t in function_type.inputs], [
        str(t) for t in function_type.results
    ]


def graph(name, graph, example):
    """The module of the graph `name`: the imported graph, <name>_graph, and
    <name>, which takes the packed parameters, then the inputs in the order
    of the module's forward, of which `example` are values. No argument is
    written."""
    # Dynamo orders the inputs of the graph as the forward first reads them.
    order = [
        next(i for i, value in enumerate(example) if value is input_)
        for input_ in graph._runtime_inputs_ref
    ]
    graph._func_name = f"{name}_graph"
    graph.lower_to_top_level_ir(buffer_layout="contiguous", writable=False)
    with ir.Context() as context:
        context.allow_unregistered_dialects = True
        module = ir.Module.parse(str(graph._imported_module))
        (imported,) = module.body.operations
        arguments, results = signature(imported)
        parameters = len(graph.params)
        packs = {"f32": [], "i8": []}
        for index, argument in enumerate(arguments[:parameters]):
            packs[
                str(ir.RankedTensorType(ir.Type.parse(argument)).element_type)
            ].append(index)
        sizes = {
            dtype: sum(math.prod(_shape(arguments[i])) for i in indices)
            for dtype, indices in packs.items()
        }
        readonly = "{bufferization.writable = false}"
        header = [
            f"%f32: tensor<{sizes['f32']}xf32> {readonly}",
            f"%i8: tensor<{sizes['i8']}xi8> {readonly}",
        ]
        inputs = [None] * len(order)
        for position, index in enumerate(order):
            inputs[index] = arguments[parameters + position]
        header += [f"%input{i}: {t} {readonly}" for i, t in enumerate(inputs)]
        body = []
        offsets = {"f32": 0, "i8": 0}
        values = [None] * parameters
        for dtype, indices in packs.items():
            for index in indices:
                tensor = arguments[index]
                shape = _shape(tensor)
                count = math.prod(shape)
                body.append(
                    f"%slice{index} = tensor.extract_slice %{dtype}[{offsets[dtype]}] [{count}] [1]"
                    f" : tensor<{sizes[dtype]}x{dtype}> to tensor<{count}x{dtype}>"
                )
                values[index] = f"%slice{index}"
                if len(shape) > 1:
                    body.append(
                        f"%param{index} = tensor.expand_shape %slice{index}"
                        f" [{list(range(len(shape)))}] output_shape {shape}"
                        f" : tensor<{count}x{dtype}> into {tensor}"
                    )
                    values[index] = f"%param{index}"
                elif not shape:
                    body.append(
                        f"%param{index} = tensor.collapse_shape %slice{index} []"
                        f" : tensor<{count}x{dtype}> into {tensor}"
                    )
                    values[index] = f"%param{index}"
                offsets[dtype] += count
        operands = values + [f"%input{index}" for index in order]
        returned = ", ".join(results)
        body.append(
            f"%results:{len(results)} = func.call @{name}_graph({', '.join(operands)})"
            f" : ({', '.join(arguments)}) -> ({returned})"
        )
        body.append(
            f"return {', '.join(f'%results#{i}' for i in range(len(results)))} : {returned}"
        )
        wrapper = ir.Module.parse(
            f"func.func private @{name}_graph({', '.join(arguments)}) -> ({returned})\n"
            f"func.func @{name}({', '.join(header)}) -> ({returned}) {{\n  "
            + "\n  ".join(body)
            + "\n}\n"
        )
        function = wrapper.body.operations[1]
        function.detach_from_parent()
        module.body.append(function)
        return str(module), signature(function)


def _shape(tensor):
    return [int(d) for d in ir.RankedTensorType(ir.Type.parse(tensor)).shape]


def prefill(vision, cameras, prefix, prefix_inputs):
    """The prefill program, from the signature of the vision function
    (`graph`) and the partition of the prefill graph (a partition.Device,
    whose graph takes the inputs named `prefix_inputs` in the order of its
    forward): `prefill_program`, on the control hart. It takes the packs of
    vision and of the prefill, the images, the prompt tokens and their mask,
    and returns what the prefill graph does."""
    with ir.Context() as context:
        context.allow_unregistered_dialects = True
        vision_arguments, vision_results = vision
        lines = prefix.declarations()
        lines.append(
            f"func.func private @vision({', '.join(vision_arguments)}) -> ({', '.join(vision_results)})"
        )
        image = vision_arguments[2]
        image_shape = _shape(image)
        images = f"tensor<{cameras}x" + image.removeprefix("tensor<")
        tokens_type, masks_type = [
            tensor_type(prefix.meta[name]) for name in prefix_inputs[:2]
        ]
        # The program writes none of its inputs.
        readonly = "{bufferization.writable = false}"
        inputs = [
            f"%vision_{dtype}: {vision_arguments[index]} {readonly}"
            for index, dtype in enumerate(("f32", "i8"))
        ]
        inputs += [
            f"%prefix_{element}: tensor<{prefix.packs[element]}x{element}> {readonly}"
            for element in ("f32", "i8")
        ]
        inputs += [
            f"%images: {images} {readonly}",
            f"%tokens: {tokens_type} {readonly}",
            f"%masks: {masks_type} {readonly}",
        ]
        body = []
        zeros = ", ".join("0" for _ in image_shape)
        ones = ", ".join("1" for _ in image_shape)
        sizes = ", ".join(str(d) for d in image_shape)
        (image_tokens,) = vision_results
        for c in range(cameras):
            body.append(
                f"%image{c} = tensor.extract_slice %images[{c}, {zeros}] [1, {sizes}] [1, {ones}]"
                f" : {images} to {image}"
            )
            body.append(f"%tokens{c}_out = tensor.empty() : {image_tokens}")
            body.append(
                f'%tokens{c} = "device.launch"(%vision_f32, %vision_i8, %image{c}, %tokens{c}_out)'
                f' <{{callee = @vision, hart_class = "dense",'
                f" operandSegmentSizes = array<i32: {len(vision_arguments)}, 1>}}>"
                f" : ({', '.join(vision_arguments + [image_tokens])}) -> ({image_tokens})"
            )
        launched, outputs = prefix.launches(
            "p_",
            {element: f"%prefix_{element}" for element in ("f32", "i8")},
            dict(
                zip(
                    prefix_inputs,
                    ["%tokens", "%masks", *[f"%tokens{c}" for c in range(cameras)]],
                )
            ),
        )
        body += launched
        returned = [tensor_type(prefix.meta[name]) for name in prefix.output.args]
        body.append(f"return {', '.join(outputs)} : {', '.join(returned)}")
        lines.append(
            f"func.func @prefill_program({', '.join(inputs)}) -> ({', '.join(returned)}) {{\n  "
            + "\n  ".join(body)
            + "\n}"
        )
        return str(ir.Module.parse("\n".join(lines)))
