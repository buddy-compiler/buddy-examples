"""Expose the existing typed decision as aligned GVN input tensors."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from models.models.laya.model import inputs

output = Path(sys.argv[1])
data = (output / "input.data").read_bytes()
cursor = 0
sizes = []
with (output / "sample/inputs.data").open("wb") as stream:
    for tensor, axes in zip(inputs(), [(1,), (1,), (), (), ()], strict=True):
        count = tensor.numel() * tensor.element_size()
        stream.write(data[cursor : cursor + count] + bytes(-count % 64))
        cursor += count
        dimensions = "".join(f" {tensor.shape[axis]}" for axis in axes)
        sizes.append(f"input {count}{dimensions}\n")
assert cursor == len(data)
sizes.extend(("output 64\n", "output 8\n"))
(output / "sample/sizes.txt").write_text("".join(sizes))
