#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

try:
  import tomllib
except ModuleNotFoundError:
  import tomli as tomllib


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


def path_key(id_path: list[int]) -> str:
  return "-".join(str(item) for item in id_path)


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
    extra_keys = set(node) - {"node", "id", "tag"}
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
    traces.append({
      "id": trace_id,
      "id_path": id_path,
      "node": node["node"],
      "tag": node["tag"],
    })
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


def read_cycle(path: Path) -> dict[str, int]:
  if not path.is_file():
    raise FileNotFoundError(f"missing cycle trace file: {path}")
  text = path.read_text(encoding="utf-8").strip()
  if not text:
    raise ValueError(f"empty cycle trace file: {path}")

  lines = text.splitlines()
  if len(lines) == 1 and len(lines[0].split()) == 1:
    try:
      elapsed = int(lines[0])
    except ValueError as exc:
      raise ValueError(f"invalid cycle trace file: {path}") from exc
    if elapsed < 0:
      raise ValueError(f"negative cycle value in: {path}")
    return {"elapsed": elapsed}

  cycle: dict[str, int] = {}
  for line in lines:
    parts = line.split()
    if len(parts) != 2:
      raise ValueError(f"invalid cycle trace line in {path}: {line}")
    key, value = parts
    if key not in ("start", "end", "elapsed"):
      raise ValueError(f"unknown cycle trace key in {path}: {key}")
    try:
      cycle[key] = int(value)
    except ValueError as exc:
      raise ValueError(f"invalid cycle trace value in {path}: {line}") from exc
    if cycle[key] < 0:
      raise ValueError(f"negative cycle trace value in {path}: {line}")

  if "elapsed" not in cycle:
    raise ValueError(f"cycle trace missing elapsed value: {path}")
  if ("start" in cycle) != ("end" in cycle):
    raise ValueError(f"cycle trace must contain both start and end: {path}")
  if "start" in cycle and cycle["end"] - cycle["start"] != cycle["elapsed"]:
    raise ValueError(f"cycle trace elapsed does not match start/end: {path}")
  return cycle


def count_lines(path: Path) -> int | None:
  if not path.is_file():
    return None
  count = 0
  with path.open("r", encoding="utf-8") as file:
    for count, _ in enumerate(file, start=1):
      pass
  return count


def collect_cycle_paths(trace_dir: Path) -> dict[tuple[int, ...], Path]:
  cycle_dir = trace_dir / "cycle"
  if not cycle_dir.is_dir():
    raise NotADirectoryError(f"missing cycle trace directory: {cycle_dir}")

  result = {}
  for path in sorted(cycle_dir.iterdir()):
    match = TRACE_FILE_RE.match(path.name)
    if not match:
      continue
    id_path = tuple(int(item) for item in match.group(1).split("-"))
    if id_path in result:
      raise ValueError(f"duplicate trace file id_path: {list(id_path)}")
    result[id_path] = path
  return result


def validate_cycle_paths(cycle_paths: dict[tuple[int, ...], Path]) -> None:
  if not cycle_paths:
    raise ValueError("no cycle trace files found")

  for id_path in sorted(cycle_paths):
    if len(id_path) == 1:
      continue
    for depth in range(1, len(id_path)):
      parent = id_path[:depth]
      if parent not in cycle_paths:
        expected = f"trace-{path_key(list(parent))}.txt"
        raise ValueError(
            f"missing parent trace file for id_path {list(id_path)}: "
            f"expected {expected}")


def trace_level(trace: dict) -> int:
  if "level" in trace:
    return trace["level"]
  return len(trace["id_path"]) - 1


def level_name(level: int) -> str:
  if level == 0:
    return "L0 graph"
  if level == 1:
    return "L1 linalg"
  if level == 2:
    return "L2 buckyball"
  return f"L{level} trace"


def build_perfetto(trace_dir: Path, trace_toml: Path,
                   mlir_paths: list[Path]) -> dict:
  traces = load_trace_config(trace_toml)
  trace_by_path = {tuple(trace["id_path"]): trace for trace in traces}
  cycle_paths = collect_cycle_paths(trace_dir)
  validate_cycle_paths(cycle_paths)
  has_nested_trace = any(len(id_path) > 1 for id_path in cycle_paths)
  if has_nested_trace and not mlir_paths:
    raise ValueError("multi-level trace output requires at least one --mlir file")
  for id_path, trace in load_trace_metadata(mlir_paths).items():
    if id_path not in trace_by_path:
      trace_by_path[id_path] = trace
  entries = []
  base_ts = None
  serial_ts = 0

  for id_path, cycle_path in sorted(cycle_paths.items()):
    trace = trace_by_path.get(id_path)
    if trace is None:
      raise ValueError(f"missing trace metadata for id_path: {list(id_path)}")
    tensor_path = trace_dir / "tensor" / f"trace-{path_key(trace['id_path'])}.txt"
    cycle = read_cycle(cycle_path)
    if "start" in cycle:
      base_ts = cycle["start"] if base_ts is None else min(base_ts, cycle["start"])
    entries.append((trace, cycle_path, tensor_path, cycle))

  events = []
  levels = sorted({trace_level(trace) for trace, _, _, _ in entries})
  for level in levels:
    events.append({
      "name": "thread_name",
      "ph": "M",
      "pid": 0,
      "tid": level,
      "args": {
        "name": level_name(level),
      },
    })

  for trace, cycle_path, tensor_path, cycle in entries:
    trace_id = trace["id"]
    id_path = trace["id_path"]
    level = trace_level(trace)
    dur = cycle["elapsed"]
    if base_ts is not None and "start" in cycle:
      ts = cycle["start"] - base_ts
    else:
      ts = serial_ts
      serial_ts += dur
    tensor_elements = count_lines(tensor_path)

    args = {
      "id": trace_id,
      "id_path": id_path,
      "node": trace["node"],
      "cycle_file": str(cycle_path),
      "elapsed_cycle": dur,
    }
    if "start" in cycle:
      args["start_cycle"] = cycle["start"]
      args["end_cycle"] = cycle["end"]
    if tensor_elements is not None:
      args["tensor_file"] = str(tensor_path)
      args["tensor_elements"] = tensor_elements
    if "trace_type" in trace:
      args["trace_type"] = trace["trace_type"]
    if "level" in trace:
      args["level"] = trace["level"]
    if "parent" in trace:
      args["parent"] = trace["parent"]

    events.append({
      "name": trace["tag"],
      "cat": "buddy.trace",
      "ph": "X",
      "ts": ts,
      "dur": dur,
      "pid": 0,
      "tid": level,
      "args": args,
    })

  return {
    "displayTimeUnit": "ns",
    "traceEvents": events,
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Convert Buddy trace output to Perfetto JSON.")
  parser.add_argument("trace_dir", type=Path,
                      help="Trace output directory containing cycle/ and tensor/.")
  parser.add_argument("trace_toml", type=Path,
                      help="trace.toml used to generate the trace output.")
  parser.add_argument("-o", "--output", type=Path,
                      help="Output Perfetto JSON path. Defaults to TRACE_DIR/perfetto.json.")
  parser.add_argument("--mlir", action="append", type=Path, default=[],
                      help="Expanded trace MLIR file. Required for multi-level trace output.")
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
