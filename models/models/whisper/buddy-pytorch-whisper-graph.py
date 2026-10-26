from pathlib import Path
import torch
from buddy.compiler.frontend import DynamoCompiler
from stack.compiler.frontend.ops import tosa
from torch._inductor.decomposition import decompositions

output = Path(__file__).resolve().parent / "output"
output.mkdir(exist_ok=True)

checkpoint = output / "whisper"
from transformers import WhisperForConditionalGeneration


class Logits(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = source

    def forward(self, audio, tokens):
        return self.source(
            input_features=audio, decoder_input_ids=tokens, use_cache=False
        ).logits


model = Logits(WhisperForConditionalGeneration.from_pretrained(checkpoint)).eval()
inputs = (torch.zeros((1, 80, 3000)), torch.zeros((1, 448), dtype=torch.int64))

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
