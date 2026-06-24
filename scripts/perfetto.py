#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

try:
  import tomllib
except ModuleNotFoundError:
  import tomli as tomllib


def load_trace_config(path: Path) -> list[dict]:
  data = tomllib.loads(path.read_text(encoding="utf-8"))
  traces = data.get("trace")
  if not isinstance(traces, list) or not traces:
    raise ValueError("trace config must contain [[trace]] entries")

  used_ids: set[int] = set()
  for trace in traces:
    if not isinstance(trace, dict):
      raise ValueError("each trace entry must be a table")
    for key in ("node", "id", "tag"):
      if key not in trace:
        raise ValueError(f"trace entry missing `{key}`")
    if not isinstance(trace["node"], str) or not trace["node"]:
      raise ValueError("trace node must be a non-empty string")
    if not isinstance(trace["tag"], str) or not trace["tag"]:
      raise ValueError("trace tag must be a non-empty string")
    if not isinstance(trace["id"], int) or trace["id"] < 0:
      raise ValueError("trace id must be a non-negative integer")
    if trace["id"] in used_ids:
      raise ValueError(f"duplicate trace id: {trace['id']}")
    used_ids.add(trace["id"])
  return traces


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


def build_perfetto(trace_dir: Path, trace_toml: Path) -> dict:
  traces = load_trace_config(trace_toml)
  entries = []
  base_ts = None
  serial_ts = 0

  for trace in sorted(traces, key=lambda item: item["id"]):
    trace_id = trace["id"]
    cycle_path = trace_dir / "cycle" / f"trace-{trace_id}.txt"
    tensor_path = trace_dir / "tensor" / f"trace-{trace_id}.txt"
    cycle = read_cycle(cycle_path)
    if "start" in cycle:
      base_ts = cycle["start"] if base_ts is None else min(base_ts, cycle["start"])
    entries.append((trace, cycle_path, tensor_path, cycle))

  events = []

  for trace, cycle_path, tensor_path, cycle in entries:
    trace_id = trace["id"]
    dur = cycle["elapsed"]
    if base_ts is not None and "start" in cycle:
      ts = cycle["start"] - base_ts
    else:
      ts = serial_ts
      serial_ts += dur
    tensor_elements = count_lines(tensor_path)

    args = {
      "id": trace_id,
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

    events.append({
      "name": trace["tag"],
      "cat": "buddy.trace",
      "ph": "X",
      "ts": ts,
      "dur": dur,
      "pid": 0,
      "tid": "cycle",
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
  data = build_perfetto(trace_dir, trace_toml)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(data, indent=2), encoding="utf-8")
  print(output)


if __name__ == "__main__":
  main()
