from pathlib import Path

try:
  import tomllib
except ModuleNotFoundError:
  import tomli as tomllib


def load_trace_config(path: Path) -> dict:
  if not path.exists():
    raise FileNotFoundError(f"trace config not found: {path}")
  data = tomllib.loads(path.read_text(encoding="utf-8"))
  items = data.get("trace")
  if not isinstance(items, list):
    raise ValueError("trace config must contain [[trace]] entries")

  result = {}
  ids = set()
  for item in items:
    extra_keys = set(item) - {"node", "id", "tag"}
    if extra_keys:
      names = ", ".join(sorted(extra_keys))
      raise ValueError(f"unsupported trace fields: {names}")
    node = item.get("node")
    trace_id = item.get("id")
    tag = item.get("tag")
    if not isinstance(node, str) or not node:
      raise ValueError("trace.node must be a non-empty string")
    if not isinstance(trace_id, int):
      raise ValueError(f"trace id for {node} must be an integer")
    if not isinstance(tag, str) or not tag:
      raise ValueError(f"trace tag for {node} must be a non-empty string")
    if node in result:
      raise ValueError(f"duplicate trace node: {node}")
    if trace_id in ids:
      raise ValueError(f"duplicate trace id: {trace_id}")
    ids.add(trace_id)
    result[node] = {"id": trace_id, "tag": tag}
  return result
