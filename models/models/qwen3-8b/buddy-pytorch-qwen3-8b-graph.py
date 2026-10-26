from pathlib import Path
import torch
from buddy.compiler.frontend import DynamoCompiler
from stack.compiler.frontend.ops import tosa
from torch._inductor.decomposition import decompositions

output = Path(__file__).resolve().parent / "output"
output.mkdir(exist_ok=True)

checkpoint = output / "qwen3-8b"
from transformers import AutoModelForCausalLM

source = AutoModelForCausalLM.from_pretrained(
    checkpoint, dtype=torch.float32, attn_implementation="eager"
)


class Logits(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = source

    def forward(self, tokens, mask):
        return self.source(
            input_ids=tokens, attention_mask=mask, use_cache=False
        ).logits


model = Logits(source).eval()
inputs = (
    torch.zeros((1, 32), dtype=torch.int64),
    torch.ones((1, 32), dtype=torch.int64),
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
