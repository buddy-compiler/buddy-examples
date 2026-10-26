from pathlib import Path
import json

from safetensors.torch import load_file
import torch
from torch._inductor.decomposition import decompositions
from transformers import AutoConfig

from buddy.compiler.frontend import DynamoCompiler
from stack.compiler.frontend.ops import tosa
from stack.models.models.laya.model import Model
from frontend.permute.graph import permute

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
model_source = model_dir.parents[3] / "models/laya"
checkpoint = model_source / "output/laya"


class Encoder(torch.nn.Module):
    """Expose ModernBERT masks as tensor operations at the import boundary."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        positions = torch.arange(input_ids.shape[1])
        visible = attention_mask[:, None, None, :].bool()
        local = (positions[:, None] - positions[None, :]).abs()
        sliding = visible & (local <= self.model.config.sliding_window)
        mask = torch.where(visible, 0.0, torch.finfo(torch.float32).min)
        sliding_mask = torch.where(sliding, 0.0, torch.finfo(torch.float32).min)
        return self.model(
            input_ids=input_ids,
            attention_mask={"full_attention": mask, "sliding_attention": sliding_mask},
        )


# ===----------------------------------------------------------------------===
# Step 1: Import the complete checkpoint model.
# ===----------------------------------------------------------------------===
settings = json.loads((checkpoint / "rl_agent_config.json").read_text())
model = Model(settings, AutoConfig.from_pretrained(checkpoint / "encoder"))
model.load_state_dict(load_file(checkpoint / "model.safetensors"), strict=True)
model.eval()
names = {
    value.data_ptr(): name
    for name, value in [*model.named_parameters(), *model.named_buffers()]
}
model.encoder = Encoder(model.encoder).eval()
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
    trace=None,
)
with torch.no_grad():
    (graph,) = compiler.importer(model, *inputs)
params = compiler.imported_params[graph]
parameter_names = {
    node.name: names[value.data_ptr()]
    for node, value in zip(graph.params, params, strict=True)
}

# ===----------------------------------------------------------------------===
# Step 2.1: Export the original graph.
# ===----------------------------------------------------------------------===
graph.dump(output / "01-import.txt")
# graph.dump_parameters(output / "01-import-parameters")

# ===----------------------------------------------------------------------===
# Step 2.2: Absorb the Linear weight transposes without changing parameters.
# ===----------------------------------------------------------------------===
permute(graph)
graph.dump(output / "02-permuted.txt")
# graph.dump_parameters(output / "02-permuted-parameters")
