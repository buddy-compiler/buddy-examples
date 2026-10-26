from safetensors.torch import load_file
from transformers import AutoConfig
from pathlib import Path
import torch
from buddy.compiler.frontend import DynamoCompiler
from stack.compiler.frontend.ops import tosa
from torch._inductor.decomposition import decompositions

output = Path(__file__).resolve().parent / "output"
output.mkdir(exist_ok=True)
import json

from model import Model

model_directory = output / "laya"
settings = json.loads((model_directory / "rl_agent_config.json").read_text())
model = Model(settings, AutoConfig.from_pretrained(model_directory / "encoder"))
model.load_state_dict(load_file(model_directory / "model.safetensors"), strict=True)
model.eval()
inputs = (
    torch.zeros((1, 128), dtype=torch.int64),
    torch.ones((1, 128), dtype=torch.int64),
    torch.arange(16, dtype=torch.int64)[None, :],
    torch.ones((1, 16), dtype=torch.bool),
    torch.zeros((1,), dtype=torch.int64),
)

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
