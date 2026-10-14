"""Route a forward graph's subgraph call through the core runtime."""

from pathlib import Path

from stack.runtime.scripts.chip_targets import load_chip, signature, target_thread


def _arguments(forward: str, symbol: str) -> int:
    head = f"func.func private @{symbol}("
    begin = index = forward.index(head) + len(head)
    arguments, depth = [], 0
    while depth or forward[index] != ")":
        char = forward[index]
        if char in "(<[":
            depth += 1
        elif char in ")>]":
            depth -= 1
        elif char == "," and not depth:
            arguments.append(forward[begin:index].strip())
            begin = index + 1
        index += 1
    arguments.append(forward[begin:index].strip())
    result = forward[index + 1 :].split("\n", 1)[0]
    if not all(argument.startswith("memref<") for argument in arguments) or (
        result.startswith(" -> ") and not result.lstrip(" ->(").startswith("memref<")
    ):
        raise ValueError(f"{symbol}: task arguments and results must be memrefs")
    return len(arguments) + result.startswith(" -> ")


def dispatch(forward: str, symbol: str, chip_pb: Path, target: str, path: Path) -> str:
    """Write the task shim for symbol and return forward calling it instead."""
    kind = signature(target_thread(load_chip(chip_pb), target))
    names = [f"a{index}" for index in range(_arguments(forward, symbol))]
    params = ", ".join(f"void *{name}" for name in names)
    path.write_text(
        "#include <runtime.h>\n\n"
        f"void _mlir_ciface_{symbol}({', '.join('void *' for _ in names)});\n\n"
        "static void run(void *frame) {\n"
        "  void **a = frame;\n"
        f"  _mlir_ciface_{symbol}({', '.join(f'a[{i}]' for i in range(len(names)))});\n"
        "}\n\n"
        f"void _mlir_ciface_task_{symbol}({params}) {{\n"
        f"  void *frame[] = {{{', '.join(names)}}};\n"
        f"  task_run({kind}ULL, run, frame);\n"
        "}\n"
    )
    return forward.replace(f"@{symbol}(", f"@task_{symbol}(")
