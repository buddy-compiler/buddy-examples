from pathlib import Path


def export_wrapper(symbol, typed_path, llvm_path, directory):
    from buddy_mlir import ir

    def descriptor(type):
        if not isinstance(type, ir.MemRefType):
            raise ValueError(
                f"{symbol}: non-MemRef CIF result/input requires an explicit ABI: {type}"
            )
        type = ir.MemRefType(type)
        if type.rank > 4 or any(size <= 0 for size in type.shape):
            raise ValueError(f"{symbol}: BufferView requires static rank <= 4")
        cpp = {"f32": "float", "i8": "int8_t", "i64": "int64_t"}.get(
            str(type.element_type)
        )
        if cpp is None:
            raise ValueError(
                f"{symbol}: unsupported CIF element type {type.element_type}"
            )
        strides, offset = type.get_strides_and_offset()
        strides = [None if ir.ShapedType.is_dynamic_size(s) else s for s in strides]
        return {
            "shape": list(type.shape),
            "strides": strides,
            "cpp": cpp,
            "rank": type.rank,
            "ctype": f"StridedMemRefType<{cpp}, {type.rank}>",
        }

    with ir.Context() as context:
        context.allow_unregistered_dialects = True
        module = ir.Module.parse(Path(typed_path).read_text())
        function = next(
            op
            for op in module.body.operations
            if op.operation.name == "func.func"
            and ir.StringAttr(op.attributes["sym_name"]).value == symbol
        )
        signature = ir.FunctionType(
            ir.TypeAttr(function.attributes["function_type"]).value
        )
        inputs = [descriptor(t) for t in signature.inputs]
        outputs = [descriptor(t) for t in signature.results]
        block = function.regions[0].blocks[0]
        return_op = list(block.operations)[-1].operation
        if return_op.name != "func.return" or len(return_op.operands) != len(outputs):
            raise ValueError(
                f"{symbol}: typed return does not match the actual CIF results"
            )
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
        for result, output in zip(return_op.operands, outputs):
            while result not in block.arguments:
                owner = result.owner.operation
                if owner.name == "memref.alloc":
                    output["ownership"] = "workspace"
                    break
                if owner.name not in views:
                    raise ValueError(
                        f"{symbol}: cannot prove output ownership through {owner.name}"
                    )
                result = owner.operands[0]
            else:
                output["ownership"] = "borrowed"
                output["alias_input"] = list(block.arguments).index(result)
        final = ir.Module.parse(Path(llvm_path).read_text())
        cif = next(
            op
            for op in final.body.operations
            if op.operation.name == "llvm.func"
            and ir.StringAttr(op.attributes["sym_name"]).value
            == "_mlir_ciface_" + symbol
        )
        arguments = list(cif.regions[0].blocks[0].arguments)
        if len(arguments) != len(inputs) + 1 or any(
            str(arg.type) != "!llvm.ptr" for arg in arguments
        ):
            raise ValueError(
                f"{symbol}: final CIF signature differs from the inspected MemRef ABI"
            )
        returned = list(cif.regions[0].blocks[0].operations)[-1].operation
        if returned.name != "llvm.return" or returned.operands:
            raise ValueError(
                f"{symbol}: final CIF must return through its explicit result pointer"
            )
        calls = [
            op.operation
            for op in cif.regions[0].blocks[0].operations
            if op.operation.name == "llvm.call"
            and "callee" in op.attributes
            and ir.FlatSymbolRefAttr(op.attributes["callee"]).value == symbol
        ]

        def llvm_descriptor(out):
            fields = "ptr,ptr,i64"
            if out["rank"]:
                fields += f',array<{out["rank"]}xi64>,array<{out["rank"]}xi64>'
            return "!llvm.struct<(" + fields + ")>"

        expected = (
            llvm_descriptor(outputs[0])
            if len(outputs) == 1
            else (
                "!llvm.struct<("
                + ",".join(
                    llvm_descriptor(out).removeprefix("!llvm.") for out in outputs
                )
                + ")>"
            )
        )
        if (
            len(calls) != 1
            or len(calls[0].results) != 1
            or "".join(str(calls[0].results[0].type).split()) != expected
        ):
            raise ValueError(
                f"{symbol}: final CIF return struct does not match every inspected descriptor field"
            )

    result_name = f"Result_{symbol}"
    prototype = f'extern "C" void _mlir_ciface_{symbol}({result_name} *'
    prototype += "".join(f", {arg['ctype']} *" for arg in inputs) + ");\n"
    header = '#pragma once\n#include "task_api.h"\n#include <CRunnerUtils.h>\n'
    header += (
        f"struct {result_name} {{\n"
        + "".join(
            f"  {out['ctype']} output{index};\n" for index, out in enumerate(outputs)
        )
        + "};\n"
    )
    header += prototype + f"void invoke_{symbol}(const BufferView *, BufferView *);\n"
    Path(directory, symbol + ".h").write_text(header)
    cpp = f'#include "{symbol}.h"\n#include <cstdlib>\n'
    cpp += f"void invoke_{symbol}(const BufferView *input, BufferView *output) {{\n"
    for index, arg in enumerate(inputs):
        cpp += f"  {arg['ctype']} arg{index}{{}};\n"
        cpp += f"  arg{index}.basePtr = static_cast<{arg['cpp']} *>(input[{index}].allocated);\n"
        cpp += (
            f"  arg{index}.data = static_cast<{arg['cpp']} *>(input[{index}].data);\n"
        )
        cpp += f"  arg{index}.offset = input[{index}].offset;\n"
        for axis in reversed(range(arg["rank"])):
            size = arg["shape"][axis]
            condition = f"input[{index}].sizes[{axis}] != {size}"
            if arg["strides"][axis] is not None:
                condition += (
                    f" || input[{index}].strides[{axis}] != {arg['strides'][axis]}"
                )
            cpp += f"  if ({condition}) std::abort();\n"
            cpp += f"  arg{index}.sizes[{axis}] = input[{index}].sizes[{axis}];\n"
            cpp += f"  arg{index}.strides[{axis}] = input[{index}].strides[{axis}];\n"
    cpp += f"  {result_name} result;\n  _mlir_ciface_{symbol}(&result"
    cpp += "".join(f", &arg{index}" for index in range(len(inputs))) + ");\n"
    for index, out in enumerate(outputs):
        cpp += f"  output[{index}] = {{}};\n"
        for dest, source in (
            ("allocated", "basePtr"),
            ("data", "data"),
            ("offset", "offset"),
        ):
            cpp += f"  output[{index}].{dest} = result.output{index}.{source};\n"
        for axis in range(out["rank"]):
            cpp += f"  output[{index}].sizes[{axis}] = result.output{index}.sizes[{axis}];\n"
            cpp += f"  output[{index}].strides[{axis}] = result.output{index}.strides[{axis}];\n"
    cpp += "}\n"
    Path(directory, symbol + ".cpp").write_text(cpp)
    return {"inputs": inputs, "outputs": outputs}


def write_index(directory, kernels):
    directory = Path(directory)
    symbols = [kernel["symbol"] for kernel in kernels]
    max_inputs = max(len(kernel["inputs"]) for kernel in kernels)
    max_outputs = max(len(kernel["outputs"]) for kernel in kernels)
    max_rank = max(
        t["rank"] for kernel in kernels for t in kernel["inputs"] + kernel["outputs"]
    )
    (directory / "task_api.h").write_text(
        "#pragma once\n#include <cstdint>\n#include <cstddef>\n"
        "struct BufferView { void *allocated; void *data; int64_t offset; int64_t sizes[4]; int64_t strides[4]; };\n"
        "using TaskEntry = void (*)(const BufferView *, BufferView *);\n"
        "extern const TaskEntry GemmaTaskEntries[];\nextern const size_t GemmaTaskEntryCount;\n"
        "extern const uint64_t GemmaTaskSignatures[];\n"
        "extern const size_t GemmaTaskInputCounts[];\nextern const size_t GemmaTaskOutputCounts[];\n"
        f"inline constexpr size_t GemmaTaskMaxInputs = {max_inputs};\n"
        f"inline constexpr size_t GemmaTaskMaxOutputs = {max_outputs};\n"
        f"inline constexpr size_t GemmaTaskMaxRank = {max_rank};\n"
    )
    source = "".join(f'#include "{symbol}.h"\n' for symbol in symbols)
    source += "const TaskEntry GemmaTaskEntries[] = {\n"
    source += "".join(f"  invoke_{symbol},\n" for symbol in symbols) + "};\n"
    source += f"const size_t GemmaTaskEntryCount = {len(symbols)};\n"
    source += (
        "const uint64_t GemmaTaskSignatures[] = {\n"
        + "".join(f"  {kernel['core_signature']}ULL,\n" for kernel in kernels)
        + "};\n"
    )
    for label in ("Input", "Output"):
        source += (
            f"const size_t GemmaTask{label}Counts[] = {{\n"
            + "".join(f"  {len(kernel[label.lower() + 's'])},\n" for kernel in kernels)
            + "};\n"
        )
    (directory / "task_index.cpp").write_text(source)
