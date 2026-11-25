"""The program of pi0 on piplus: the vision encoder once per camera, the
prefix once, the denoising step once per step with the Euler update after it.

`run` computes it in PyTorch. `graph` exports one graph as an MLIR module and
`program` the program as a device program: a function of the control hart
that launches vision on a dense hart once per camera, the prefix and every
step as the launches of their partitions (compiler.frontend.partition: the
ranks on the action and dense harts), and computes the Euler updates
itself. Placement on the harts and the transfers between them are the
device compiler's.

A graph takes its parameters as two packed buffers, FP32 and INT8, in the
order of its parameters, the prefix and the step those of their partitions;
the program takes those of vision, the prefix and the step, then the inputs
of the policy, and returns the actions."""

import math
import struct

import torch

from buddy_mlir import ir
from compiler.frontend.partition import tensor_type


def run(vision, prefix, step, steps, images, tokens, token_masks, state, noise):
    image_tokens = [vision(image) for image in images.unbind(0)]
    masks, angles, *cache = prefix(tokens, token_masks, *image_tokens)
    x_t = noise
    for k in range(steps):
        time = torch.full((1,), 1.0 - k / steps)
        x_t = x_t + step(x_t, time, state, masks, angles, *cache) * (-1.0 / steps)
    return x_t


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


def _f32(value):
    """An FP32 constant exactly: its bits."""
    return "0x%08X" % struct.unpack("<I", struct.pack("<f", value))[0]


def program(vision, steps, cameras, prefix, prefix_inputs, step, step_inputs):
    """The device program, from the signature of the vision function
    (`graph`) and the partitions of the prefix and the step (each a
    partition.Device, whose graph takes the inputs named `prefix_inputs`,
    `step_inputs` in the order of its forward): `program`, on the control
    hart."""
    with ir.Context() as context:
        context.allow_unregistered_dialects = True
        vision_arguments, vision_results = vision
        lines = prefix.declarations() + step.declarations()
        lines.append(
            f"func.func private @vision({', '.join(vision_arguments)}) -> ({', '.join(vision_results)})"
        )
        image = vision_arguments[2]
        image_shape = _shape(image)
        images = f"tensor<{cameras}x" + image.removeprefix("tensor<")
        lang_tokens, lang_masks = [
            tensor_type(prefix.meta[name]) for name in prefix_inputs[:2]
        ]
        actions, time, state = [
            tensor_type(step.meta[name]) for name in step_inputs[:3]
        ]
        (velocity,) = [tensor_type(step.meta[name]) for name in step.output.args]
        # The program writes none of its inputs.
        readonly = "{bufferization.writable = false}"
        inputs = [
            f"%vision_{dtype}: {vision_arguments[index]} {readonly}"
            for index, dtype in enumerate(("f32", "i8"))
        ]
        for name, device in (("prefix", prefix), ("step", step)):
            inputs += [
                f"%{name}_{element}: tensor<{device.packs[element]}x{element}> {readonly}"
                for element in ("f32", "i8")
            ]
        inputs += [
            f"%images: {images} {readonly}",
            f"%lang_tokens: {lang_tokens} {readonly}",
            f"%lang_masks: {lang_masks} {readonly}",
            f"%state: {state} {readonly}",
            f"%noise: {actions} {readonly}",
        ]
        body = []
        zeros = ", ".join("0" for _ in image_shape)
        ones = ", ".join("1" for _ in image_shape)
        sizes = ", ".join(str(d) for d in image_shape)
        (tokens_type,) = vision_results
        for c in range(cameras):
            body.append(
                f"%image{c} = tensor.extract_slice %images[{c}, {zeros}] [1, {sizes}] [1, {ones}]"
                f" : {images} to {image}"
            )
            body.append(f"%tokens{c}_out = tensor.empty() : {tokens_type}")
            body.append(
                f'%tokens{c} = "device.launch"(%vision_f32, %vision_i8, %image{c}, %tokens{c}_out)'
                f' <{{callee = @vision, hart_class = "dense",'
                f" operandSegmentSizes = array<i32: {len(vision_arguments)}, 1>}}>"
                f" : ({', '.join(vision_arguments + [tokens_type])}) -> ({tokens_type})"
            )
        tokens = [f"%tokens{c}" for c in range(cameras)]
        launched, outputs = prefix.launches(
            "p_",
            {element: f"%prefix_{element}" for element in ("f32", "i8")},
            dict(zip(prefix_inputs, ["%lang_tokens", "%lang_masks", *tokens])),
        )
        body += launched
        body.append(f"%rate = arith.constant dense<{_f32(-1.0 / steps)}> : {velocity}")
        x = "%noise"
        for k in range(steps):
            body.append(
                f"%time{k} = arith.constant dense<{_f32(1.0 - k / steps)}> : {time}"
            )
            launched, (result,) = step.launches(
                f"s{k}_",
                {element: f"%step_{element}" for element in ("f32", "i8")},
                dict(zip(step_inputs, [x, f"%time{k}", "%state", *outputs])),
            )
            body += launched
            body.append(f"%change{k} = arith.mulf {result}, %rate : {velocity}")
            body.append(f"%x{k + 1} = arith.addf {x}, %change{k} : {actions}")
            x = f"%x{k + 1}"
        body.append(f"return {x} : {actions}")
        lines.append(
            f"func.func @program({', '.join(inputs)}) -> {actions} {{\n  "
            + "\n  ".join(body)
            + "\n}"
        )
        return str(ir.Module.parse("\n".join(lines)))
