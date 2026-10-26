from pathlib import Path
import torch
from buddy.compiler.frontend import DynamoCompiler
from stack.compiler.frontend.ops import tosa
from torch._inductor.decomposition import decompositions

output = Path(__file__).resolve().parent / "output"
output.mkdir(exist_ok=True)

checkpoint = output / "mlp-mixer"
from safetensors.torch import load_file
from model import Mixer

model = Mixer().eval()
state = load_file(Path(checkpoint) / "model.safetensors")
state["stem.proj.weight"] = state["stem.proj.weight"].flatten(1)
model.load_state_dict(state, strict=True)
inputs = (torch.zeros((196, 768)),)

compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=decompositions,
    verbose=True,
    verbose_path=output / "buddy-graph.txt",
    trace=None,
)
with torch.no_grad():
    graphs = compiler.importer(model, *inputs)
for graph in graphs:
    graph.lower_to_top_level_ir()
