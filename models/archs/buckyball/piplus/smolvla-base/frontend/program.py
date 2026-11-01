"""The program of SmolVLA on piplus: the vision encoder once per camera, the
prefix once, the denoising step once per step with the Euler update after it.

`run` computes it in PyTorch. `graph` exports one graph as an MLIR module and
`program` the program as a device program: a function of the control core
that launches each graph on its class of core — vision and prefix on a dense
core, the steps on the action core — and computes the Euler updates itself.
Placement on the cores and the transfers between them are the device
compiler's.

A graph takes its parameters as two packed buffers, FP32 and INT8, in the
order of its parameters; the program takes those of every graph, then the
inputs of the policy, and returns the actions."""

import math
import struct

import torch

from buddy_mlir import ir

classes = {"vision": "dense", "prefix": "dense", "step": "action"}


def run(vision, prefix, step, steps, images, lang_tokens, lang_masks, state, noise):
    tokens = [vision(image) for image in images.unbind(0)]
    pad_masks, *cache = prefix(lang_tokens, lang_masks, state, *tokens)
    x_t = noise
    for k in range(steps):
        time = torch.full((1,), 1.0 - k / steps)
        x_t = x_t + step(x_t, time, pad_masks, *cache) * (-1.0 / steps)
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
                body.append(
                    f"%shape{index} = arith.constant dense<{shape}> : tensor<{len(shape)}xindex>"
                    if shape
                    else f"%shape{index} = tensor.empty() : tensor<0xindex>"
                )
                body.append(
                    f"%param{index} = tensor.reshape %slice{index}(%shape{index})"
                    f" : (tensor<{count}x{dtype}>, tensor<{len(shape)}xindex>) -> {tensor}"
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


def program(signatures, steps, cameras):
    """The device program, from the signatures of the graph functions
    (`graph`): `program`, on the control core."""
    with ir.Context() as context:
        context.allow_unregistered_dialects = True
        lines = []
        for name, (arguments, results) in signatures.items():
            lines.append(
                f"func.func private @{name}({', '.join(arguments)}) -> ({', '.join(results)})"
            )
        vision_arguments, vision_results = signatures["vision"]
        prefix_arguments, prefix_results = signatures["prefix"]
        step_arguments, step_results = signatures["step"]
        image = vision_arguments[2]
        image_shape = _shape(image)
        images = f"tensor<{cameras}x" + image.removeprefix("tensor<")
        lang_tokens, lang_masks, state = prefix_arguments[2:5]
        actions, time = step_arguments[2:4]
        (velocity,) = step_results
        # The program writes none of its inputs.
        readonly = "{bufferization.writable = false}"
        inputs = [
            f"%{name}_{dtype}: {arguments[index]} {readonly}"
            for name, (arguments, _) in signatures.items()
            for index, dtype in enumerate(("f32", "i8"))
        ]
        inputs += [
            f"%images: {images} {readonly}",
            f"%lang_tokens: {lang_tokens} {readonly}",
            f"%lang_masks: {lang_masks} {readonly}",
            f"%state: {state} {readonly}",
            f"%noise: {actions} {readonly}",
        ]
        body = []

        def launch(result, name, operands, results):
            arguments = signatures[name][0]
            inits = []
            for index, type_ in enumerate(results):
                body.append(f"%{result}_out{index} = tensor.empty() : {type_}")
                inits.append(f"%{result}_out{index}")
            packed = [f"%{name}_f32", f"%{name}_i8"]
            many = f":{len(results)}" if len(results) > 1 else ""
            body.append(
                f'%{result}{many} = "device.launch"({", ".join(packed + operands + inits)})'
                f' <{{callee = @{name}, core_class = "{classes[name]}",'
                f" operandSegmentSizes = array<i32: {len(arguments)}, {len(results)}>}}>"
                f" : ({', '.join(arguments + results)}) -> ({', '.join(results)})"
            )

        zeros = ", ".join("0" for _ in image_shape)
        ones = ", ".join("1" for _ in image_shape)
        sizes = ", ".join(str(d) for d in image_shape)
        for c in range(cameras):
            body.append(
                f"%image{c} = tensor.extract_slice %images[{c}, {zeros}] [1, {sizes}] [1, {ones}]"
                f" : {images} to {image}"
            )
            launch(f"tokens{c}", "vision", [f"%image{c}"], vision_results)
        tokens = [f"%tokens{c}" for c in range(cameras)]
        launch(
            "prefix",
            "prefix",
            ["%lang_tokens", "%lang_masks", "%state", *tokens],
            prefix_results,
        )
        prefix = [f"%prefix#{i}" for i in range(len(prefix_results))]
        body.append(f"%rate = arith.constant dense<{_f32(-1.0 / steps)}> : {velocity}")
        x = "%noise"
        for k in range(steps):
            body.append(
                f"%time{k} = arith.constant dense<{_f32(1.0 - k / steps)}> : {time}"
            )
            launch(f"velocity{k}", "step", [x, f"%time{k}", *prefix], step_results)
            body.append(f"%change{k} = arith.mulf %velocity{k}, %rate : {velocity}")
            body.append(f"%x{k + 1} = arith.addf {x}, %change{k} : {actions}")
            x = f"%x{k + 1}"
        body.append(f"return {x} : {actions}")
        lines.append(
            f"func.func @program({', '.join(inputs)}) -> {actions} {{\n  "
            + "\n  ".join(body)
            + "\n}"
        )
        return str(ir.Module.parse("\n".join(lines)))


def step_program(signatures):
    """One denoising step alone as a device program: `step_program`, on the
    control core, launches the step graph once on an action core with the
    inputs it takes (the actions, the time, the prefix padding mask and its
    keys and values) and returns the velocity."""
    with ir.Context() as context:
        context.allow_unregistered_dialects = True
        arguments, results = signatures["step"]
        readonly = "{bufferization.writable = false}"
        names = [f"%argument{i}" for i in range(len(arguments))]
        (velocity,) = results
        body = [
            f"%velocity_out = tensor.empty() : {velocity}",
            f'%velocity = "device.launch"({", ".join(names + ["%velocity_out"])})'
            f' <{{callee = @step, core_class = "{classes["step"]}",'
            f" operandSegmentSizes = array<i32: {len(arguments)}, 1>}}>"
            f" : ({', '.join(arguments + results)}) -> ({velocity})",
            f"return %velocity : {velocity}",
        ]
        inputs = [
            f"{name}: {type_} {readonly}" for name, type_ in zip(names, arguments)
        ]
        return str(
            ir.Module.parse(
                f"func.func private @step({', '.join(arguments)}) -> ({velocity})\n"
                f"func.func @step_program({', '.join(inputs)}) -> {velocity} {{\n  "
                + "\n  ".join(body)
                + "\n}"
            )
        )
