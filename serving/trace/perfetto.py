#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import tomllib

TRACE_FILE_RE = re.compile(r"^trace-(\d+(?:-\d+)*)\.txt$")
ID_PATH_RE = re.compile(r"id_path = \[([0-9,\s]+)\]")
TAG_RE = re.compile(r'tag = "([^"]+)"')
TRACE_TYPE_RE = re.compile(r'trace_type = "([^"]+)"')
LEVEL_RE = re.compile(r"level = ([0-9]+) : i64")
PARENT_RE = re.compile(r"parent = ([0-9]+) : i64")


def parse_id(value: object) -> tuple[int, list[int]]:
    if isinstance(value, int) and value >= 0:
        return value, [value]
    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, int) and item >= 0 for item in value)
    ):
        return value[0], list(value)
    raise ValueError("trace.node id must be a non-negative integer or integer list")


def parse_int_list(text: str) -> list[int]:
    result = []
    for item in text.split(","):
        value = item.strip()
        if value:
            result.append(int(value))
    if not result:
        raise ValueError("empty id_path")
    return result


def load_trace_config(path: Path) -> list[dict]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    trace_data = data.get("trace")
    if not isinstance(trace_data, dict):
        raise ValueError("trace config must contain [trace]")

    extra_keys = set(trace_data) - {"node", "extend"}
    if extra_keys:
        names = ", ".join(sorted(extra_keys))
        raise ValueError(f"unsupported trace fields: {names}")

    nodes = trace_data.get("node")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("trace config must contain [[trace.node]] entries")

    traces = []
    used_paths: set[tuple[int, ...]] = set()
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("each trace.node entry must be a table")
        extra_keys = set(node) - {"node", "id", "tag", "extend"}
        if extra_keys:
            names = ", ".join(sorted(extra_keys))
            raise ValueError(f"unsupported trace.node fields: {names}")
        for key in ("node", "id", "tag"):
            if key not in node:
                raise ValueError(f"trace.node entry missing `{key}`")
        if not isinstance(node["node"], str) or not node["node"]:
            raise ValueError("trace node must be a non-empty string")
        if not isinstance(node["tag"], str) or not node["tag"]:
            raise ValueError("trace tag must be a non-empty string")
        trace_id, id_path = parse_id(node["id"])
        path_tuple = tuple(id_path)
        if path_tuple in used_paths:
            raise ValueError(f"duplicate trace id_path: {id_path}")
        used_paths.add(path_tuple)
        traces.append(
            {
                "id": trace_id,
                "id_path": id_path,
                "node": node["node"],
                "tag": node["tag"],
            }
        )
    return traces


def parse_int_attr(line: str, pattern: re.Pattern[str]) -> int | None:
    match = pattern.search(line)
    if not match:
        return None
    return int(match.group(1))


def load_trace_metadata(paths: list[Path]) -> dict[tuple[int, ...], dict]:
    result = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"trace mlir does not exist: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if "buddy_trace.start" not in line:
                continue
            id_match = ID_PATH_RE.search(line)
            tag_match = TAG_RE.search(line)
            if not id_match or not tag_match:
                continue
            id_path = parse_int_list(id_match.group(1))
            key = tuple(id_path)
            meta = {
                "id": id_path[0],
                "id_path": id_path,
                "node": "",
                "tag": tag_match.group(1),
            }
            trace_type_match = TRACE_TYPE_RE.search(line)
            if trace_type_match:
                meta["trace_type"] = trace_type_match.group(1)
            level = parse_int_attr(line, LEVEL_RE)
            if level is not None:
                meta["level"] = level
            parent = parse_int_attr(line, PARENT_RE)
            if parent is not None:
                meta["parent"] = parent
            result[key] = meta
    return result


def read_cycle(path: Path) -> list[dict[str, int | str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing cycle trace file: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty cycle trace file: {path}")

    lines = text.splitlines()

    records = []
    cycle: dict[str, int | str] = {}
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"invalid cycle trace line in {path}: {line}")
        key, value = parts
        if key not in ("platform", "counter", "pid", "tid", "start", "end", "elapsed"):
            raise ValueError(f"unknown cycle trace key in {path}: {key}")
        if key in cycle:
            raise ValueError(f"duplicate cycle trace key in {path}: {key}")
        if key in ("platform", "counter"):
            allowed = (
                ("linux", "baremetal")
                if key == "platform"
                else ("riscv-cycle", "x86-tsc")
            )
            if value not in allowed:
                raise ValueError(f"invalid trace {key}: {value}")
            cycle[key] = value
            continue
        try:
            cycle[key] = int(value)
        except ValueError as exc:
            raise ValueError(f"invalid cycle trace value in {path}: {line}") from exc
        if cycle[key] < 0:
            raise ValueError(f"negative cycle trace value in {path}: {line}")
        if key == "elapsed":
            if set(cycle) != {
                "platform",
                "counter",
                "pid",
                "tid",
                "start",
                "end",
                "elapsed",
            }:
                raise ValueError(
                    f"cycle trace requires platform/counter/pid/tid/start/end/elapsed: {path}"
                )
            if cycle["platform"] == "linux" and (
                cycle["pid"] <= 0 or cycle["tid"] <= 0
            ):
                raise ValueError(
                    f"Linux trace requires positive process/thread IDs: {path}"
                )
            if cycle["platform"] == "baremetal" and cycle["counter"] != "riscv-cycle":
                raise ValueError(
                    f"baremetal trace requires riscv-cycle counter: {path}"
                )
            if cycle["platform"] == "baremetal" and cycle["pid"] != 0:
                raise ValueError(
                    f"baremetal trace requires pid=0 and a real hart ID: {path}"
                )
            if "start" in cycle and cycle["end"] - cycle["start"] != cycle["elapsed"]:
                raise ValueError(
                    f"cycle trace elapsed does not match start/end: {path}"
                )
            records.append(cycle)
            cycle = {}
    if cycle or not records:
        raise ValueError(f"cycle trace missing elapsed value: {path}")
    return records


def count_lines(path: Path) -> int | None:
    if not path.is_file():
        return None
    count = 0
    with path.open("r", encoding="utf-8") as file:
        for count, _ in enumerate(file, start=1):
            pass
    return count


def collect_cycle_paths(
    trace_dir: Path,
) -> dict[tuple[int, int, tuple[int, ...]], Path]:
    result = {}
    for path in sorted(trace_dir.glob("controller-*/core-*/cycle/trace-*.txt")):
        controller = re.fullmatch(r"controller-(\d+)", path.parents[2].name)
        core = re.fullmatch(r"core-(\d+)", path.parents[1].name)
        match = TRACE_FILE_RE.fullmatch(path.name)
        if not controller or not core or not match or not path.is_file():
            raise ValueError(f"invalid scoped cycle trace: {path}")
        identity = (
            int(controller[1]),
            int(core[1]),
            tuple(int(v) for v in match[1].split("-")),
        )
        if identity in result:
            raise ValueError(f"duplicate scoped trace: {identity}")
        result[identity] = path
    if not result:
        raise ValueError(f"no controller/core cycle traces: {trace_dir}")
    return result


def build_perfetto(trace_dir: Path, trace_toml: Path, mlir_paths: list[Path]) -> dict:
    trace_by_path = {tuple(t["id_path"]): t for t in load_trace_config(trace_toml)}
    for path, trace in load_trace_metadata(mlir_paths).items():
        if path not in trace_by_path:
            trace_by_path[path] = trace
    cycle_paths = collect_cycle_paths(trace_dir)
    if any(len(key[2]) > 1 for key in cycle_paths) and not mlir_paths:
        raise ValueError("multi-level trace requires trace MLIR metadata")
    entries, intervals = [], {}
    for (controller, core, path), file in cycle_paths.items():
        if path not in trace_by_path:
            raise ValueError(f"missing trace metadata for {path}")
        trace = trace_by_path[path]
        tensor = file.parent.parent / "tensor" / file.name
        records = read_cycle(file)
        for cycle in records:
            identity = (controller, core, path)
            intervals.setdefault(identity, []).append((cycle["start"], cycle["end"]))
            entries.append((controller, core, trace, file, tensor, cycle))
    if len({entry[-1]["counter"] for entry in entries}) != 1:
        raise ValueError("one trace cannot mix different counter sources")
    for directory in {entry[3].parent for entry in entries}:
        pairs = [
            line.split()
            for line in (directory / "summary.txt").read_text().splitlines()
        ]
        if any(len(pair) != 2 for pair in pairs):
            raise ValueError(f"invalid trace summary: {directory}")
        summary = dict(pairs)
        expected_keys = {
            "platform",
            "counter",
            "first_start",
            "last_end",
            "trace_span",
            "traced_cycle_sum",
            "trace_count",
        }
        if len(summary) != len(pairs) or set(summary) != expected_keys:
            raise ValueError(f"missing or duplicate trace summary fields: {directory}")
        scoped = [entry for entry in entries if entry[3].parent == directory]
        first = min(entry[-1]["start"] for entry in scoped)
        last = max(entry[-1]["end"] for entry in scoped)
        expected = {
            "first_start": first,
            "last_end": last,
            "trace_span": last - first,
            "traced_cycle_sum": sum(
                entry[-1]["elapsed"]
                for entry in scoped
                if len(entry[2]["id_path"]) == 1
            ),
            "trace_count": len(scoped),
        }
        if any(
            summary[key] != entry[-1][key]
            for entry in scoped
            for key in ("platform", "counter")
        ) or any(int(summary[key]) != value for key, value in expected.items()):
            raise ValueError(
                f"trace summary disagrees with its scoped records: {directory}"
            )
    for identity, records in intervals.items():
        records.sort()
        if any(a[1] > b[0] for a, b in zip(records, records[1:])):
            raise ValueError(f"overlapping calls in trace scope {identity}")
    for controller, core, trace, file, tensor, cycle in entries:
        path = tuple(trace["id_path"])
        if len(path) == 1:
            continue
        parents = [
            (start, end)
            for (owner, _, parent), records in intervals.items()
            if owner == controller and parent == path[:-1]
            for start, end in records
            if start <= cycle["start"] and cycle["end"] <= end
        ]
        if len(parents) != 1:
            raise ValueError(
                f"trace {controller}/{core}/{path} has {len(parents)} enclosing parents"
            )
    base = min(entry[-1]["start"] for entry in entries)
    events, threads = [], set()
    for controller, core, trace, file, tensor, cycle in entries:
        thread = (cycle["pid"], cycle["tid"])
        if thread not in threads:
            threads.add(thread)
            events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": thread[0],
                    "tid": thread[1],
                    "args": {"name": f"controller {controller}, thread {thread[1]}"},
                }
            )
        args = {
            "id": trace["id"],
            "id_path": trace["id_path"],
            "node": trace["node"],
            "platform": cycle["platform"],
            "counter": cycle["counter"],
            "controller": controller,
            "core": core,
            "cycle_file": str(file),
        }
        unit = "cycle" if cycle["counter"] == "riscv-cycle" else "tick"
        args.update(
            {
                f"start_{unit}": cycle["start"],
                f"end_{unit}": cycle["end"],
                f"elapsed_{unit}": cycle["elapsed"],
            }
        )
        tensor_elements = count_lines(tensor)
        if tensor_elements is not None:
            args.update(tensor_file=str(tensor), tensor_elements=tensor_elements)
        for key in ("trace_type", "level", "parent"):
            if key in trace:
                args[key] = trace[key]
        events.append(
            {
                "name": trace["tag"],
                "cat": "buddy.trace",
                "ph": "X",
                "ts": cycle["start"] - base,
                "dur": cycle["elapsed"],
                "pid": cycle["pid"],
                "tid": cycle["tid"],
                "args": args,
            }
        )
    return {"displayTimeUnit": "ns", "traceEvents": events}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Buddy trace output to Perfetto JSON."
    )
    parser.add_argument(
        "trace_dir", type=Path, help="Trace root containing controller-N/core-M scopes."
    )
    parser.add_argument(
        "trace_toml", type=Path, help="trace.toml used to generate the trace output."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output Perfetto JSON path. Defaults to TRACE_DIR/perfetto.json.",
    )
    parser.add_argument(
        "--mlir",
        action="append",
        type=Path,
        default=[],
        help="Expanded trace MLIR file. Required for multi-level trace output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_dir = args.trace_dir.resolve()
    trace_toml = args.trace_toml.resolve()
    if not trace_dir.is_dir():
        raise NotADirectoryError(f"trace directory does not exist: {trace_dir}")
    if not trace_toml.is_file():
        raise FileNotFoundError(f"trace.toml does not exist: {trace_toml}")

    output = args.output.resolve() if args.output else trace_dir / "perfetto.json"
    mlir_paths = [path.resolve() for path in args.mlir]
    data = build_perfetto(trace_dir, trace_toml, mlir_paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
