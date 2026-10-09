import argparse
import json
import math
import sys
from pathlib import Path


class Pool:
    def __init__(self, capacity):
        self.blocks = [[0, capacity - 48, True]]
        self.live = {}
        self.high_water = 0
        self.used = self.peak = 0

    def allocate(self, name, size):
        size = (size + 15) // 16 * 16
        for index, block in enumerate(self.blocks):
            start, available, free = block
            if not free or available < size:
                continue
            if available - size >= 64:
                self.blocks.insert(
                    index + 1, [start + 48 + size, available - size - 48, True]
                )
                block[1] = size
            block[2] = False
            self.live[name] = block
            self.used += block[1] + 48
            self.peak = max(self.peak, self.used)
            self.high_water = max(self.high_water, start + 48 + block[1])
            return
        raise ValueError("workspace allocation exceeds the planned arena")

    def free(self, name):
        block = self.live.pop(name)
        index = self.blocks.index(block)
        self.used -= block[1] + 48
        block[2] = True
        if index + 1 < len(self.blocks) and self.blocks[index + 1][2]:
            following = self.blocks.pop(index + 1)
            block[1] += 48 + following[1]
        if index and self.blocks[index - 1][2]:
            self.blocks[index - 1][1] += 48 + block[1]
            self.blocks.pop(index)


def allocations(path, name, ir):
    with ir.Context() as context, ir.Location.unknown():
        # Bank intrinsics have already passed the final LLVM NPU check. The
        # planner needs their operands, rather than their target implementation.
        context.allow_unregistered_dialects = True
        module = ir.Module.parse(path.read_text())
        functions = [
            op
            for op in module.body.operations
            if op.operation.name == "func.func"
            and ir.StringAttr(op.attributes["sym_name"]).value == name
        ]
        if len(functions) != 1 or len(functions[0].regions[0].blocks) != 1:
            raise ValueError(f"{path}: workspace requires one function block")
        block = functions[0].regions[0].blocks[0]
        events, identifiers, live = [], {}, set()
        maximum_alignment = 16
        for operation in block.operations:
            op = operation.operation
            pending = [
                child
                for region in op.regions
                for nested in region.blocks
                for child in nested.operations
            ]
            while pending:
                nested = pending.pop().operation
                if nested.name in ("memref.alloc", "memref.dealloc"):
                    raise ValueError(
                        f"{path}: workspace cannot prove nested allocation lifetimes"
                    )
                pending.extend(
                    child
                    for region in nested.regions
                    for nested_block in region.blocks
                    for child in nested_block.operations
                )
            if op.name == "memref.alloc":
                t = ir.MemRefType(op.results[0].type)
                if any(size < 0 for size in t.shape) or op.operands:
                    raise ValueError(f"{path}: workspace needs static allocations")
                if t != ir.MemRefType.get(t.shape, t.element_type):
                    raise ValueError(
                        f"{path}: workspace cannot prove a nonidentity allocation layout"
                    )
                element = t.element_type
                widths = {"i1": 1, "i8": 1, "i32": 4, "i64": 8, "f32": 4}
                if str(element) not in widths:
                    raise ValueError(
                        f"{path}: unsupported workspace element type {element}"
                    )
                width = widths[str(element)]
                alignment = (
                    ir.IntegerAttr(op.attributes["alignment"]).value
                    if "alignment" in op.attributes
                    else 0
                )
                if alignment and (alignment < 1 or alignment & (alignment - 1)):
                    raise ValueError(
                        f"{path}: workspace alignment must be a power of two"
                    )
                maximum_alignment = max(maximum_alignment, alignment)
                identifier = len(identifiers)
                identifiers[op.results[0]] = identifier
                live.add(identifier)
                # FinalizeMemRefToLLVM's malloc lowering reserves the full
                # requested alignment before adjusting the aligned pointer.
                events.append(
                    ("allocate", identifier, math.prod(t.shape) * width + alignment)
                )
            elif op.name == "memref.dealloc":
                operand = op.operands[0]
                if operand not in identifiers or identifiers[operand] not in live:
                    raise ValueError(
                        f"{path}: workspace cannot prove an aliased or duplicate free"
                    )
                identifier = identifiers[operand]
                live.remove(identifier)
                events.append(("free", identifier, 0))
            elif op.name == "func.return":
                if any(value not in identifiers for value in op.operands):
                    raise ValueError(
                        f"{path}: workspace cannot prove returned buffer ownership"
                    )
                returned = {identifiers[value] for value in op.operands}
                if returned != live:
                    raise ValueError(
                        f"{path}: workspace contains an escaping allocation"
                    )
        if not block.operations or block.operations[-1].operation.name != "func.return":
            raise ValueError(f"{path}: workspace requires an explicit return")
        return events, returned, maximum_alignment


def plan(prefill, decode):
    capacity = (
        sum(
            size + 64
            for events, _, _ in (prefill, decode)
            for action, _, size in events
            if action == "allocate"
        )
        * 513
    )
    pool = Pool(capacity)

    def execute(program, epoch):
        events, returned, _ = program
        for action, identifier, size in events:
            name = (epoch, identifier)
            if action == "allocate":
                pool.allocate(name, size)
            else:
                pool.free(name)
        return {(epoch, identifier) for identifier in returned}

    # The caller copies prefill caches into separately allocated MemRefs, then
    # frees all prefill workspace results before the first decode.
    previous = execute(prefill, 0)
    for name in previous:
        pool.free(name)
    previous = set()
    # A decode keeps its previous returned caches/logits alive until its new
    # results have returned. The maximum 512-token cache admits 511 decodes.
    for epoch in range(1, 512):
        current = execute(decode, epoch)
        for name in previous:
            pool.free(name)
        previous = current
    for name in previous:
        pool.free(name)
    if pool.live:
        raise ValueError("workspace results remain live after shutdown")
    alignment = max(prefill[2], decode[2])
    budget = (pool.high_water + 64 + alignment - 1) // alignment * alignment
    return {
        "bytes": budget,
        "peak_used_bytes": pool.peak,
        "allocation_extent_bytes": pool.high_water,
        "alignment": alignment,
        "decode_iterations": 511,
        "block_header_bytes": 48,
        "payload_alignment": 16,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler-build", type=Path, required=True)
    parser.add_argument("--prefill", type=Path, required=True)
    parser.add_argument("--decode", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.compiler_build.resolve() / "python_packages"))
    from buddy_mlir import ir
    from buddy_mlir.dialects import arith, func, memref  # noqa: F401

    result = plan(
        allocations(args.prefill, "subgraph0_prefill", ir),
        allocations(args.decode, "subgraph0_decode", ir),
    )
    metadata = json.loads(args.metadata.read_text())
    metadata["workspace_bytes"] = result["bytes"]
    metadata["workspace_plan"] = result
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    args.output.write_text(
        "#pragma once\n#include <cstddef>\ninline constexpr size_t workspaceBytes = "
        + str(result["bytes"])
        + ";\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
