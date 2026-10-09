import json
import math
from pathlib import Path


def task_memory(directory, manifest):
    from buddy_mlir import ir

    plans = {}
    for kernel in manifest["kernels"]:
        symbol = kernel["symbol"]
        with ir.Context() as context:
            context.allow_unregistered_dialects = True
            module = ir.Module.parse(
                (Path(directory) / (symbol + ".bank.mlir")).read_text()
            )
            function = next(
                op
                for op in module.body.operations
                if op.operation.name == "func.func"
                and ir.StringAttr(op.attributes["sym_name"]).value == symbol
            )
            if len(function.regions[0].blocks) != 1:
                raise ValueError(
                    "Gemma workspace planning requires straight-line task ownership"
                )
            block = function.regions[0].blocks[0]
            roots = {
                value: ("input", index) for index, value in enumerate(block.arguments)
            }
            events, returns = [], []
            views = {
                "memref.cast",
                "memref.subview",
                "memref.transpose",
                "memref.reinterpret_cast",
                "memref.expand_shape",
                "memref.collapse_shape",
                "memref.reshape",
                "memref.view",
            }
            for value in block.operations:
                op = value.operation

                def nested_alloc(operation):
                    return operation.name in ("memref.alloc", "memref.dealloc") or any(
                        nested_alloc(child.operation)
                        for region in operation.regions
                        for inner in region.blocks
                        for child in inner.operations
                    )

                if any(
                    nested_alloc(child.operation)
                    for region in op.regions
                    for inner in region.blocks
                    for child in inner.operations
                ):
                    raise ValueError(
                        "Gemma cannot plan nested or conditional allocation lifetimes"
                    )
                if op.name == "memref.alloca":
                    roots[op.results[0]] = ("stack", len(roots))
                elif op.name == "memref.alloc":
                    t = ir.MemRefType(op.results[0].type)
                    if op.operands or any(size <= 0 for size in t.shape):
                        raise ValueError(
                            "Gemma workspace allocation must have a static size"
                        )
                    alignment = (
                        ir.IntegerAttr(op.attributes["alignment"]).value
                        if "alignment" in op.attributes
                        else 0
                    )
                    bytes = (
                        math.prod(t.shape)
                        * {"i1": 1, "i8": 1, "i32": 4, "i64": 8, "f32": 4}[
                            str(t.element_type)
                        ]
                        + alignment
                    )
                    identifier = len(events)
                    roots[op.results[0]] = ("allocation", identifier)
                    events.append(("allocate", identifier, bytes))
                elif op.name in views:
                    if op.operands[0] not in roots:
                        raise ValueError(
                            f"{symbol}: workspace view has no proven allocation root: {op}"
                        )
                    roots[op.results[0]] = roots[op.operands[0]]
                elif op.name == "memref.dealloc":
                    root = roots[op.operands[0]]
                    if root[0] != "allocation":
                        raise ValueError("Gemma task frees a borrowed input")
                    events.append(("free", root[1], 0))
                elif op.name == "func.return":
                    returns = [roots[value] for value in op.operands]
                    if any(root[0] == "stack" for root in returns):
                        raise ValueError("Gemma task returns a stack allocation")
            if len(returns) != len(kernel["outputs"]):
                raise ValueError(
                    "Gemma task return ownership does not match its actual ABI"
                )
            plans[symbol] = (events, returns)
    return plans


def lifetime_plan(directory, manifest):
    from .workspace import Pool

    calls = manifest["calls"]
    last, produced, exports, last_layer_outputs = {}, set(), set(), {}

    def reads(event):
        if event["kind"] == "compute":
            return [
                binding["slot"] for binding in event["bindings"] if "slot" in binding
            ]
        if event["kind"] in ("view", "last_valid_row"):
            return [event["input"]] + ([event["valid"]] if "valid" in event else [])
        return event.get("inputs", []) + (
            [event["start"], event["count"]] if event["kind"] == "cache_update" else []
        )

    for index, event in enumerate(calls):
        for name in reads(event):
            last[name] = index
        if (
            event["kind"] in ("gather", "view")
            and event.get("stage") == "logits_gather"
        ):
            exports.add(event["output"])
        if event["kind"] == "compute" and event["stage"] == "post_layer":
            last_layer_outputs[event["phase"]] = event["outputs"]
    exports.update(
        name
        for outputs in last_layer_outputs.values()
        for name in outputs
        if name.endswith("hidden_next")
    )
    memory = task_memory(directory, manifest)
    capacity = 64 + sum(
        (size + 15) // 16 * 16 + 64
        for events, _ in memory.values()
        for action, _, size in events
        if action == "allocate"
    ) * len(calls)
    pools = [Pool(capacity) for _ in manifest["tiles"]]
    aliases, active, result = {}, {}, []
    phase = None
    for index, event in enumerate(calls):
        if event["phase"] != phase:
            for key in list(active):
                pools[active[key]].free(key)
                del active[key]
            phase = event["phase"]
            result.append(
                {
                    "kind": "keep_live",
                    "phase": phase,
                    "layer": event["layer"],
                    "slots": [],
                }
            )
        if event["kind"] == "compute":
            worker = max(0, event["rank"])
            events, returns = memory[event["kernel"]]
            for action, identifier, size in events:
                key = (worker, index, identifier)
                if action == "allocate":
                    pools[worker].allocate(key, size)
                    active[key] = worker
                else:
                    pools[worker].free(key)
                    del active[key]
            returned = {
                (worker, index, root[1]) for root in returns if root[0] == "allocation"
            }
            local_live = {key for key in active if key[0] == worker and key[1] == index}
            if local_live != returned:
                raise ValueError(
                    "Gemma task has a freed return or a non-returned live allocation"
                )
            for name, root in zip(event["outputs"], returns):
                if root[0] == "allocation":
                    aliases[name] = (worker, index, root[1])
                else:
                    binding = event["bindings"][root[1]]
                    aliases[name] = aliases.get(binding.get("slot"))
                produced.add(name)
        elif event["kind"] in ("view", "last_valid_row"):
            aliases[event["output"]] = aliases.get(event["input"])
            produced.add(event["output"])
        elif event["kind"] == "gather":
            aliases[event["output"]] = None
            produced.add(event["output"])
        result.append(event)
        boundary = (
            event["kind"] in ("gather", "cache_update")
            or (
                event["kind"] == "view"
                and "stage" in event
                and event["stage"].endswith("_gather")
            )
            or (
                event["kind"] == "compute"
                and event["stage"] in ("post_layer", "post_attention", "final_norm")
            )
        )
        if boundary:
            keep = sorted(
                name
                for name in produced
                if last.get(name, -1) > index or name in exports
            )
            retained = {aliases.get(name) for name in keep}
            for key in list(active):
                if key not in retained:
                    pools[active[key]].free(key)
                    del active[key]
            result.append(
                {
                    "kind": "keep_live",
                    "phase": event["phase"],
                    "layer": event["layer"],
                    "slots": keep,
                }
            )
    bound = max((pool.high_water + 63) // 16 * 16 for pool in pools)
    return result, bound


def layout(shape, strides=None):
    if strides is None:
        strides, size = [], 1
        for extent in reversed(shape):
            strides.insert(0, size)
            size *= extent
    return (
        "{nullptr,nullptr,0,{"
        + ",".join(map(str, shape + [0] * (4 - len(shape))))
        + "},{"
        + ",".join(map(str, strides + [0] * (4 - len(shape))))
        + "}}"
    )


def emit_program(directory, manifest, metadata, prefill):
    directory = Path(directory)
    events, bound = lifetime_plan(directory, manifest)
    manifest["calls"] = events
    manifest["workspace_live_bound_bytes"] = bound
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    kernels = {
        kernel["symbol"]: (index, kernel)
        for index, kernel in enumerate(manifest["kernels"])
    }
    slots, resources, groups = {}, {}, {}
    reserved = {"valid_tokens"}
    for phase, tokens in (("prefill", prefill), ("decode", 1)):
        slots[phase + ".tokens"] = ([1, tokens], 8)
        slots[phase + ".positions"] = ([1, tokens], 8)
        reserved.update((phase + ".tokens", phase + ".positions"))
    slots["valid_tokens"] = ([1], 8)
    for cache in range(15):
        width = 512 if cache % 5 == 4 else 256
        slots[f"cache{cache}.length"] = ([1], 8)
        reserved.add(f"cache{cache}.length")
        for name in ("key_codes", "value_codes"):
            slots[f"cache{cache}.{name}"] = ([1, 1, 512, width], 1)
            reserved.add(f"cache{cache}.{name}")
        for name in ("key_scales", "value_scales"):
            slots[f"cache{cache}.{name}"] = ([1, 1, 512, width // 32], 1)
            reserved.add(f"cache{cache}.{name}")

    def names(values):
        return "{" + ",".join(json.dumps(value) for value in values) + "}"

    for event in manifest["calls"]:
        phase, layer = event["phase"], event["layer"]
        body = groups.setdefault((phase, layer), [])
        if event["kind"] == "compute":
            index, kernel = kernels[event["kernel"]]
            bindings = []
            for bind in event["bindings"]:
                if "slot" in bind:
                    name = (
                        phase + "." + bind["slot"]
                        if bind["slot"] in ("tokens", "positions")
                        else bind["slot"]
                    )
                    bindings.append("{" + json.dumps(name) + ",nullptr,0,{}}")
                else:
                    size = metadata["parameters"][phase][
                        (
                            "weight_bytes"
                            if bind["resource"].endswith("weights.bin")
                            else "f32_elements"
                        )
                    ]
                    resources[bind["resource"]] = (
                        size if bind["resource"].endswith("weights.bin") else size * 4,
                        32768 if bind["resource"].endswith("weights.bin") else 16,
                    )
                    bindings.append(
                        "{nullptr,"
                        + json.dumps(bind["resource"])
                        + ","
                        + str(bind["byte_offset"])
                        + ","
                        + layout(bind["shape"], bind["strides"])
                        + "}"
                    )
            owned = []
            for name, result in zip(event["outputs"], kernel["outputs"]):
                slots[name] = (
                    result["shape"],
                    {"float": 4, "int8_t": 1, "int64_t": 8}[result["cpp"]],
                )
                owned.append("true" if result["ownership"] == "workspace" else "false")
            body.append(
                f"  execution.compute({event['rank']},{index},{{{','.join(bindings)}}},{names(event['outputs'])},{{{','.join(owned)}}});"
            )
        elif event["kind"] == "view":
            shape, width = slots[event["input"]]
            shape = list(shape)
            shape[event["axis"]] = event["count"]
            slots[event["output"]] = (shape, width)
            body.append(
                f"  execution.view({json.dumps(event['input'])},{json.dumps(event['output'])},{event['axis']},{event['start']},{event['count']});"
            )
        elif event["kind"] == "last_valid_row":
            shape, width = slots[event["input"]]
            shape = list(shape)
            shape[1] = 1
            slots[event["output"]] = (shape, width)
            body.append(
                f"  execution.lastValidRow({json.dumps(event['input'])},{json.dumps(event['output'])},{json.dumps(event['valid'])});"
            )
        elif event["kind"] == "gather":
            shape, width = slots[event["inputs"][0]]
            shape = list(shape)
            shape[-1] = sum(span["columns"] for span in event["columns"])
            slots[event["output"]] = (shape, width)
            reserved.add(event["output"])
            spans = (
                "{"
                + ",".join(
                    "{" + str(span["column_start"]) + "," + str(span["columns"]) + "}"
                    for span in event["columns"]
                )
                + "}"
            )
            body.append(
                f"  execution.gather({names(event['inputs'])},{spans},{json.dumps(event['output'])});"
            )
        elif event["kind"] == "cache_update":
            body.append(
                f"  execution.cacheUpdate({names(event['inputs'])},{names(event['outputs'])},{json.dumps(event['start'])},{json.dumps(event['count'])});"
            )
        elif event["kind"] == "keep_live":
            body.append(f"  execution.keepLive({names(event['slots'])});")
        else:
            raise ValueError("Unknown Gemma task event")
    declarations = [
        "#pragma once",
        '#include "layer_execution.h"',
        "inline constexpr size_t GemmaTiles[] = {"
        + ",".join(map(str, manifest["tiles"]))
        + "};",
        "std::unique_ptr<LayerExecution> createGemmaExecution(const std::filesystem::path &, const std::vector<int> &);",
        "void runPrefill(LayerExecution &);",
        "void runDecode(LayerExecution &);",
    ]
    sources = []
    for (phase, layer), body in groups.items():
        symbol = f"run_{phase}_{'entry' if layer < 0 else 'layer' + str(layer)}"
        declarations.append(f"void {symbol}(LayerExecution &);")
        filename = symbol + ".cpp"
        (directory / filename).write_text(
            '#include "gemma-plan.h"\nvoid '
            + symbol
            + "(LayerExecution &execution) {\n"
            + "\n".join(body)
            + "\n}\n"
        )
        sources.append(filename)
    index = '#include "gemma-plan.h"\n#include <vector>\n'
    for phase, name in (("prefill", "runPrefill"), ("decode", "runDecode")):
        index += f"void {name}(LayerExecution &execution) {{\n"
        index += "".join(
            f"  run_{phase}_{'entry' if layer < 0 else 'layer' + str(layer)}(execution);\n"
            for p, layer in groups
            if p == phase
        )
        index += "}\n"
    index += 'std::unique_ptr<LayerExecution> createGemmaExecution(const std::filesystem::path &directory,const std::vector<int> &cpus) {\n  if(cpus.size()!=sizeof(GemmaTiles)/sizeof(GemmaTiles[0])) throw std::runtime_error("Gemma task controller count differs from captured tiles");\n  auto execution = std::unique_ptr<LayerExecution>(new LayerExecution(directory,cpus,{\n'
    index += (
        "".join(
            "    {" + json.dumps(name) + "," + str(size) + "," + str(alignment) + "},\n"
            for name, (size, alignment) in resources.items()
        )
        + "  },{\n"
    )
    index += "".join(
        "    {"
        + json.dumps(name)
        + ","
        + layout(shape)
        + ","
        + str(len(shape))
        + ","
        + str(width)
        + (",true" if name in reserved else ",false")
        + "},\n"
        for name, (shape, width) in slots.items()
    )
    index += f"  }},{bound}));\n"
    for cache in range(15):
        width = 512 if cache % 5 == 4 else 256
        index += f"  {{ std::vector<int8_t> scale({512 * width // 32}, 127);\n"
        for name in ("key_scales", "value_scales"):
            index += f'    execution->set("cache{cache}.{name}", scale.data(), scale.size());\n'
        index += "  }\n"
    index += "  return execution;\n}\n"
    (directory / "gemma-plan.h").write_text("\n".join(declarations) + "\n")
    (directory / "plan_index.cpp").write_text(index)
    return [*sources, "plan_index.cpp"], bound
